from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from dsemachine.encoding.hardware import HardwareConfig
from dsemachine.encoding.matrix_mapping import (
    AssignedTile,
    Dataflow,
    MappingSplitNode,
    MatrixShape,
    expand_mapping,
)
from dsemachine.perfsim.engine import ExecutionEngine
from dsemachine.perfsim.instructions import Instruction, InstructionKind, InstructionTrace
from dsemachine.perfsim.resources import (
    DRAMResource,
    HardwareResource,
    IOReductionResource,
    InputD2DResource,
    MatrixComputeResource,
    OutputD2DResource,
)


@dataclass(frozen=True)
class MatrixSimResult:
    total_cycles: int
    input_d2d_cycles: int
    output_d2d_cycles: int
    dram_cycles: int
    compute_cycles: int
    reduction_cycles: int
    instruction_trace: tuple[InstructionTrace, ...]

    def save_perf_trace(
        self,
        path: str | Path,
        *,
        ns_per_cycle: float = 1.0,
        module_name: str = "DSEMachine perfsim",
        display_time_unit: str = "ns",
    ) -> None:
        from dsemachine.perfsim.perf_trace import save_perfetto_trace

        save_perfetto_trace(
            self.instruction_trace,
            path,
            ns_per_cycle=ns_per_cycle,
            module_name=module_name,
            display_time_unit=display_time_unit,
        )


def simulate_matrix(
    shape: MatrixShape,
    hw: HardwareConfig,
    mapping: MappingSplitNode,
    dataflow: Dataflow,
) -> MatrixSimResult:
    if mapping.parent is not None:
        raise ValueError("root mapping node must not have a parent")
    if mapping.tile_ids_from_parent is not None:
        raise ValueError("root mapping node must not have tile_ids_from_parent")
    if mapping.rect.k_size != shape.k or mapping.rect.n_size != shape.n:
        raise ValueError("root mapping rect must cover the full matrix K/N shape")

    assigned_tiles = expand_mapping(mapping, hw)
    resources = _build_resources(hw)
    instructions = _build_tile_instructions(shape, assigned_tiles, dataflow)
    instructions.extend(_build_reduction_instructions(shape, assigned_tiles))

    engine_result = ExecutionEngine(resources).run(instructions)
    return MatrixSimResult(
        total_cycles=engine_result.total_cycles,
        input_d2d_cycles=_sum_cycles(engine_result.trace, InstructionKind.INPUT_D2D),
        output_d2d_cycles=_sum_cycles(engine_result.trace, InstructionKind.OUTPUT_D2D),
        dram_cycles=_sum_cycles(engine_result.trace, InstructionKind.DRAM_LOAD),
        compute_cycles=_sum_cycles(engine_result.trace, InstructionKind.COMPUTE),
        reduction_cycles=_sum_cycles(engine_result.trace, InstructionKind.IO_REDUCTION),
        instruction_trace=engine_result.trace,
    )


def _build_resources(hw: HardwareConfig) -> dict[str, HardwareResource]:
    resources: dict[str, HardwareResource] = {
        "io.reduction": IOReductionResource("io.reduction", hw),
    }
    for chiplet_id in range(hw.num_pim_chiplets):
        prefix = f"pim{chiplet_id}"
        resources[f"{prefix}.input_d2d"] = InputD2DResource(f"{prefix}.input_d2d", hw)
        resources[f"{prefix}.output_d2d"] = OutputD2DResource(f"{prefix}.output_d2d", hw)
        resources[f"{prefix}.dram"] = DRAMResource(f"{prefix}.dram", hw)
        resources[f"{prefix}.compute"] = MatrixComputeResource(f"{prefix}.compute", hw)
    return resources


def _build_tile_instructions(
    shape: MatrixShape,
    assigned_tiles: list[AssignedTile],
    dataflow: Dataflow,
) -> list[Instruction]:
    instructions: list[Instruction] = []
    for index, assigned in enumerate(assigned_tiles):
        prefix = f"tile{index}"
        chiplet = f"pim{assigned.chiplet_id}"
        payload = {
            "shape": shape,
            "tile": assigned,
            "dataflow": dataflow,
        }

        input_id = f"{prefix}.input_d2d"
        dram_id = f"{prefix}.dram_load"
        compute_id = f"{prefix}.compute"
        output_id = f"{prefix}.output_d2d"

        instructions.append(
            Instruction(
                id=input_id,
                kind=InstructionKind.INPUT_D2D,
                resource_name=f"{chiplet}.input_d2d",
                deps=(),
                payload=payload,
            )
        )
        instructions.append(
            Instruction(
                id=dram_id,
                kind=InstructionKind.DRAM_LOAD,
                resource_name=f"{chiplet}.dram",
                deps=(input_id,),
                payload=payload,
            )
        )
        instructions.append(
            Instruction(
                id=compute_id,
                kind=InstructionKind.COMPUTE,
                resource_name=f"{chiplet}.compute",
                deps=(input_id,),
                payload=payload,
            )
        )
        instructions.append(
            Instruction(
                id=output_id,
                kind=InstructionKind.OUTPUT_D2D,
                resource_name=f"{chiplet}.output_d2d",
                deps=(dram_id, compute_id),
                payload=payload,
            )
        )
    return instructions


def _build_reduction_instructions(
    shape: MatrixShape,
    assigned_tiles: list[AssignedTile],
) -> list[Instruction]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, assigned in enumerate(assigned_tiles):
        groups[(assigned.tile.n_offset, assigned.tile.n_size)].append(index)

    instructions: list[Instruction] = []
    for reduction_index, ((n_start, n_size), tile_indices) in enumerate(groups.items()):
        if len(tile_indices) <= 1:
            continue
        deps = tuple(f"tile{tile_index}.output_d2d" for tile_index in tile_indices)
        instructions.append(
            Instruction(
                id=f"reduction{reduction_index}.io",
                kind=InstructionKind.IO_REDUCTION,
                resource_name="io.reduction",
                deps=deps,
                payload={
                    "shape": shape,
                    "n_start": n_start,
                    "n_size": n_size,
                    "num_partials": len(tile_indices),
                },
            )
        )
    return instructions


def _sum_cycles(trace: tuple[InstructionTrace, ...], kind: InstructionKind) -> int:
    return sum(item.end - item.start for item in trace if item.kind is kind)
