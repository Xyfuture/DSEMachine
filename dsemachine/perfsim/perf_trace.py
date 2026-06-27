from __future__ import annotations

from pathlib import Path

from dsemachine.perfsim.instructions import InstructionTrace


def save_perfetto_trace(
    trace: tuple[InstructionTrace, ...],
    path: str | Path,
    *,
    ns_per_cycle: float = 1.0,
    module_name: str = "DSEMachine perfsim",
    display_time_unit: str = "ns",
) -> None:
    from perf_tracer import PerfettoTracer

    tracer = PerfettoTracer(ns_per_cycle=ns_per_cycle)
    module = tracer.register_module(module_name)
    tracks = {
        resource_name: tracer.register_track(resource_name, module)
        for resource_name in sorted({item.resource_name for item in trace})
    }

    for item in trace:
        tracer.complete_event(
            tracks[item.resource_name],
            start_ts=item.start,
            end_ts=item.end,
            name=item.inst_id,
            category=item.kind.value,
        )

    tracer.save(str(path), display_time_unit=display_time_unit)
