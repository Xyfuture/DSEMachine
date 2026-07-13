from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


POWER_LIMIT_W = 650.0
AREA_LIMIT_MM2 = 950.0
TPOT_LIMIT_MS = 18.0


@dataclass(frozen=True)
class MockHardware:
    num_pim_chiplets: int
    pe_m: int
    pe_n: int
    num_sa: int
    sa_m: int
    sa_n: int
    d2d_bw_gbps: int
    dram_bw_gbps: int
    io_sram_mb: int
    process_node: str


@dataclass(frozen=True)
class MockMappingResult:
    tile_k: int
    tile_n: int
    utilization: float
    mapping_latency_ms: float
    io_buffer_peak_mb: float


@dataclass(frozen=True)
class MockPPACResult:
    tokens_per_dollar: float
    tpot_ms: float
    energy_per_token_j: float
    power_w: float
    area_mm2: float
    cost_usd: float
    tokens_per_second: float


PROCESS_NODE_TABLE = {
    "12nm": {"perf": 0.75, "power": 0.72, "area": 1.35, "cost": 0.55},
    "7nm": {"perf": 1.00, "power": 1.00, "area": 1.00, "cost": 1.00},
    "5nm": {"perf": 1.18, "power": 1.14, "area": 0.82, "cost": 1.45},
    "3nm": {"perf": 1.34, "power": 1.30, "area": 0.68, "cost": 2.10},
}


def sample_hardware(trial: optuna.Trial) -> MockHardware:
    return MockHardware(
        num_pim_chiplets=trial.suggest_categorical("num_pim_chiplets", [4, 8, 12, 16, 24, 32]),
        pe_m=trial.suggest_categorical("pe_m", [2, 4, 8]),
        pe_n=trial.suggest_categorical("pe_n", [2, 4, 8]),
        num_sa=trial.suggest_categorical("num_sa", [1, 2, 4]),
        sa_m=trial.suggest_categorical("sa_m", [8, 16, 32]),
        sa_n=trial.suggest_categorical("sa_n", [8, 16, 32]),
        d2d_bw_gbps=trial.suggest_categorical("d2d_bw_gbps", [256, 512, 768, 1024, 1536]),
        dram_bw_gbps=trial.suggest_categorical("dram_bw_gbps", [512, 1024, 1536, 2048, 3072]),
        io_sram_mb=trial.suggest_categorical("io_sram_mb", [64, 128, 192, 256, 384, 512]),
        process_node=trial.suggest_categorical("process_node", list(PROCESS_NODE_TABLE)),
    )


def mock_mapping_search(hw: MockHardware) -> MockMappingResult:
    compute_parallelism = hw.num_pim_chiplets * hw.pe_m * hw.pe_n * hw.num_sa
    memory_balance = min(hw.dram_bw_gbps / max(1.0, compute_parallelism * 24.0), 1.25)
    d2d_balance = min(hw.d2d_bw_gbps / max(1.0, hw.num_pim_chiplets * 54.0), 1.20)

    utilization = 0.50 + 0.23 * memory_balance + 0.20 * d2d_balance
    utilization -= 0.010 * abs(hw.pe_m - hw.pe_n)
    utilization = max(0.35, min(utilization, 0.96))

    tile_k = 64 * max(1, hw.sa_m // 8)
    tile_n = 64 * max(1, hw.sa_n // 8)
    mapping_latency_ms = 7.0 / math.sqrt(max(1.0, compute_parallelism * utilization))
    io_buffer_peak_mb = 48.0 + 3.8 * hw.num_pim_chiplets + 0.020 * hw.d2d_bw_gbps

    return MockMappingResult(
        tile_k=tile_k,
        tile_n=tile_n,
        utilization=utilization,
        mapping_latency_ms=mapping_latency_ms,
        io_buffer_peak_mb=io_buffer_peak_mb,
    )


def mock_simulate_ppac(hw: MockHardware, mapping: MockMappingResult) -> MockPPACResult:
    process = PROCESS_NODE_TABLE[hw.process_node]
    macs = hw.num_pim_chiplets * hw.pe_m * hw.pe_n * hw.num_sa * hw.sa_m * hw.sa_n
    bandwidth_score = math.sqrt(hw.d2d_bw_gbps * hw.dram_bw_gbps) / 1024.0
    perf_score = macs * mapping.utilization * bandwidth_score * process["perf"]

    sram_shortage = max(0.0, mapping.io_buffer_peak_mb - hw.io_sram_mb)
    sram_penalty = 1.0 + sram_shortage / 180.0
    tpot_ms = (42_000.0 / max(perf_score, 1.0) + mapping.mapping_latency_ms) * sram_penalty

    pe_count = hw.num_pim_chiplets * hw.pe_m * hw.pe_n
    area_mm2 = (
        52.0
        + process["area"] * (0.030 * macs + 2.2 * pe_count)
        + 0.085 * hw.io_sram_mb
        + 0.018 * (hw.d2d_bw_gbps + hw.dram_bw_gbps)
    )
    power_w = (
        38.0
        + process["power"] * (0.016 * macs + 1.3 * pe_count)
        + 0.030 * hw.io_sram_mb
        + 0.045 * hw.d2d_bw_gbps
        + 0.030 * hw.dram_bw_gbps
    )
    cost_usd = (
        process["cost"] * area_mm2 * 10.5
        + hw.num_pim_chiplets * 22.0
        + hw.io_sram_mb * 0.35
        + 140.0
    )

    tokens_per_second = 1000.0 / max(tpot_ms, 0.001)
    energy_per_token_j = power_w / max(tokens_per_second, 0.001)
    tokens_per_dollar = tokens_per_second / max(cost_usd, 0.001)

    return MockPPACResult(
        tokens_per_dollar=tokens_per_dollar,
        tpot_ms=tpot_ms,
        energy_per_token_j=energy_per_token_j,
        power_w=power_w,
        area_mm2=area_mm2,
        cost_usd=cost_usd,
        tokens_per_second=tokens_per_second,
    )


def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    hw = sample_hardware(trial)
    mapping = mock_mapping_search(hw)
    ppac = mock_simulate_ppac(hw, mapping)

    for key, value in asdict(hw).items():
        trial.set_user_attr(f"hw_{key}", value)
    for key, value in asdict(mapping).items():
        trial.set_user_attr(f"mapping_{key}", value)
    for key, value in asdict(ppac).items():
        trial.set_user_attr(f"ppac_{key}", value)

    trial.set_user_attr("constraint_power_w", ppac.power_w - POWER_LIMIT_W)
    trial.set_user_attr("constraint_area_mm2", ppac.area_mm2 - AREA_LIMIT_MM2)
    trial.set_user_attr("constraint_sram_mb", mapping.io_buffer_peak_mb - hw.io_sram_mb)
    trial.set_user_attr("constraint_tpot_ms", ppac.tpot_ms - TPOT_LIMIT_MS)

    return ppac.tokens_per_dollar, ppac.tpot_ms, ppac.energy_per_token_j


def constraints_func(trial: FrozenTrial) -> Sequence[float]:
    return (
        float(trial.user_attrs.get("constraint_power_w", 1.0)),
        float(trial.user_attrs.get("constraint_area_mm2", 1.0)),
        float(trial.user_attrs.get("constraint_sram_mb", 1.0)),
        float(trial.user_attrs.get("constraint_tpot_ms", 1.0)),
    )


def export_trials(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_trials = [trial for trial in study.trials if trial.state is TrialState.COMPLETE]

    base_fields = ["number", "state", "values"]
    param_fields = sorted({key for trial in completed_trials for key in trial.params})
    attr_fields = sorted({key for trial in completed_trials for key in trial.user_attrs})
    fieldnames = base_fields + [f"param_{key}" for key in param_fields] + attr_fields

    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for trial in completed_trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "values": list(trial.values or ()),
            }
            row.update({f"param_{key}": trial.params.get(key, "") for key in param_fields})
            row.update({key: trial.user_attrs.get(key, "") for key in attr_fields})
            writer.writerow(row)

    pareto_numbers = {trial.number for trial in study.best_trials}
    with (output_dir / "pareto_front.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for trial in completed_trials:
            if trial.number in pareto_numbers:
                row = {
                    "number": trial.number,
                    "state": trial.state.name,
                    "values": list(trial.values or ()),
                }
                row.update({f"param_{key}": trial.params.get(key, "") for key in param_fields})
                row.update({key: trial.user_attrs.get(key, "") for key in attr_fields})
                writer.writerow(row)


def print_summary(study: optuna.Study) -> None:
    completed = [trial for trial in study.trials if trial.state is TrialState.COMPLETE]
    print(f"Completed trials: {len(completed)}")
    print(f"Pareto trials: {len(study.best_trials)}")
    print()
    for trial in sorted(study.best_trials, key=lambda item: item.number)[:10]:
        values = trial.values or [0.0, 0.0, 0.0]
        feasible = all(value <= 0 for value in constraints_func(trial))
        print(
            "Trial "
            f"{trial.number:03d} | feasible={feasible} | "
            f"tokens/$={values[0]:.6f} | "
            f"TPOT={values[1]:.3f} ms | "
            f"energy/token={values[2]:.3f} J | "
            f"chiplets={trial.params.get('num_pim_chiplets')} | "
            f"node={trial.params.get('process_node')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock multi-objective Optuna TPESampler demo for a DSE loop.",
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=50,
        seed=args.seed,
        constraints_func=constraints_func,
    )
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        sampler=sampler,
        study_name="mock_dse_multi_objective_tpe",
    )

    study.optimize(objective, n_trials=args.n_trials)
    export_trials(study, args.output_dir)
    print_summary(study)
    print()
    print(f"Wrote results to: {args.output_dir}")


if __name__ == "__main__":
    main()
