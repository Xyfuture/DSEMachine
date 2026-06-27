from __future__ import annotations

from dataclasses import dataclass

from dsemachine.perfsim.instructions import Instruction, InstructionTrace
from dsemachine.perfsim.resources import HardwareResource


@dataclass(frozen=True)
class EngineResult:
    total_cycles: int
    trace: tuple[InstructionTrace, ...]


class ExecutionEngine:
    def __init__(self, resources: dict[str, HardwareResource]) -> None:
        if not resources:
            raise ValueError("engine requires at least one resource")
        self.resources = resources

    def run(self, instructions: list[Instruction]) -> EngineResult:
        if not instructions:
            raise ValueError("engine requires at least one instruction")

        instructions_by_id = {inst.id: inst for inst in instructions}
        if len(instructions_by_id) != len(instructions):
            raise ValueError("instruction ids must be unique")

        ordered = self._topological_sort(instructions_by_id)
        end_times: dict[str, int] = {}
        traces: list[InstructionTrace] = []

        for inst in ordered:
            if inst.resource_name not in self.resources:
                raise ValueError(f"unknown resource: {inst.resource_name}")
            ready_time = max((end_times[dep] for dep in inst.deps), default=0)
            start, end = self.resources[inst.resource_name].reserve(ready_time, inst)
            end_times[inst.id] = end
            traces.append(
                InstructionTrace(
                    inst_id=inst.id,
                    kind=inst.kind,
                    resource_name=inst.resource_name,
                    start=start,
                    end=end,
                    deps=inst.deps,
                )
            )

        total_cycles = max(
            max(end_times.values()),
            max(resource.free_time for resource in self.resources.values()),
        )
        return EngineResult(total_cycles=total_cycles, trace=tuple(traces))

    def _topological_sort(self, instructions_by_id: dict[str, Instruction]) -> list[Instruction]:
        state: dict[str, int] = {}
        ordered: list[Instruction] = []

        def visit(inst_id: str) -> None:
            if inst_id not in instructions_by_id:
                raise ValueError(f"unknown dependency: {inst_id}")
            current = state.get(inst_id, 0)
            if current == 1:
                raise ValueError("instruction graph contains a cycle")
            if current == 2:
                return

            state[inst_id] = 1
            inst = instructions_by_id[inst_id]
            for dep in inst.deps:
                visit(dep)
            state[inst_id] = 2
            ordered.append(inst)

        for inst_id in instructions_by_id:
            visit(inst_id)
        return ordered
