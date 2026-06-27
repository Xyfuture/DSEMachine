from __future__ import annotations

from dataclasses import dataclass

from dsemachine.encoding.hardware import HardwareConfig
from dsemachine.encoding.matrix_mapping import AssignedTile, Dataflow, MatrixShape, ceil_div
from dsemachine.perfsim.instructions import Instruction


@dataclass
class HardwareResource:
    name: str
    hw: HardwareConfig
    free_time: int = 0

    def estimate_cycles(self, instruction: Instruction) -> int:
        raise NotImplementedError

    def reserve(self, ready_time: int, instruction: Instruction) -> tuple[int, int]:
        start = max(self.free_time, ready_time)
        end = start + self.estimate_cycles(instruction)
        self.free_time = end
        return start, end


class InputD2DResource(HardwareResource):
    def estimate_cycles(self, instruction: Instruction) -> int:
        shape = _shape(instruction)
        tile = _tile(instruction).tile
        bytes_count = shape.m * tile.k_size * shape.input_bytes
        return ceil_div(bytes_count, self.hw.d2d_input_bytes_per_cycle)


class OutputD2DResource(HardwareResource):
    def estimate_cycles(self, instruction: Instruction) -> int:
        shape = _shape(instruction)
        tile = _tile(instruction).tile
        bytes_count = shape.m * tile.n_size * shape.accumulator_bytes
        return ceil_div(bytes_count, self.hw.d2d_output_bytes_per_cycle)


class DRAMResource(HardwareResource):
    def estimate_cycles(self, instruction: Instruction) -> int:
        shape = _shape(instruction)
        tile = _tile(instruction).tile
        bytes_count = tile.k_size * tile.n_size * shape.weight_bytes
        aligned = ceil_div(bytes_count, self.hw.dram_page_bytes) * self.hw.dram_page_bytes
        return ceil_div(aligned, self.hw.dram_bytes_per_cycle)


class MatrixComputeResource(HardwareResource):
    def estimate_cycles(self, instruction: Instruction) -> int:
        shape = _shape(instruction)
        tile = _tile(instruction).tile
        dataflow = _dataflow(instruction)
        if dataflow is Dataflow.OS:
            return (
                ceil_div(shape.m, self.hw.sa_m)
                * ceil_div(tile.k_size, self.hw.pe_m * self.hw.num_sa)
                * ceil_div(tile.n_size, self.hw.pe_n * self.hw.sa_n)
            )
        if dataflow is Dataflow.WS:
            return (
                shape.m
                * ceil_div(tile.k_size, self.hw.pe_m * self.hw.num_sa * self.hw.sa_m)
                * ceil_div(tile.n_size, self.hw.pe_n * self.hw.sa_n)
            )
        if dataflow is Dataflow.IS:
            return (
                tile.n_size
                * ceil_div(shape.m, self.hw.pe_n * self.hw.sa_n)
                * ceil_div(tile.k_size, self.hw.pe_m * self.hw.num_sa * self.hw.sa_m)
            )
        raise ValueError(f"unknown dataflow: {dataflow}")


class IOReductionResource(HardwareResource):
    def estimate_cycles(self, instruction: Instruction) -> int:
        shape = _shape(instruction)
        n_size = instruction.payload["n_size"]
        num_partials = instruction.payload["num_partials"]
        bytes_count = num_partials * shape.m * n_size * shape.accumulator_bytes
        return ceil_div(bytes_count, self.hw.io_reduction_bytes_per_cycle)


def _shape(instruction: Instruction) -> MatrixShape:
    value = instruction.payload["shape"]
    if not isinstance(value, MatrixShape):
        raise ValueError("instruction shape payload must be MatrixShape")
    return value


def _tile(instruction: Instruction) -> AssignedTile:
    value = instruction.payload["tile"]
    if not isinstance(value, AssignedTile):
        raise ValueError("instruction tile payload must be AssignedTile")
    return value


def _dataflow(instruction: Instruction) -> Dataflow:
    value = instruction.payload["dataflow"]
    if not isinstance(value, Dataflow):
        raise ValueError("instruction dataflow payload must be Dataflow")
    return value
