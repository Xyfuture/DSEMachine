from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    num_pim_chiplets: int
    pe_m: int
    pe_n: int
    num_sa: int
    sa_m: int
    sa_n: int
    sa_frequency_mhz: int
    dram_bytes_per_cycle: int
    d2d_input_bytes_per_cycle: int
    d2d_output_bytes_per_cycle: int
    io_reduction_bytes_per_cycle: int
    dram_page_bytes: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
