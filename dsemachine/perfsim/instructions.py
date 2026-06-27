from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class InstructionKind(Enum):
    INPUT_D2D = "input_d2d"
    DRAM_LOAD = "dram_load"
    COMPUTE = "compute"
    OUTPUT_D2D = "output_d2d"
    IO_REDUCTION = "io_reduction"


@dataclass(frozen=True)
class Instruction:
    id: str
    kind: InstructionKind
    resource_name: str
    deps: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class InstructionTrace:
    inst_id: str
    kind: InstructionKind
    resource_name: str
    start: int
    end: int
    deps: tuple[str, ...]
