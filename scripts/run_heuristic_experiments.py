"""Reproducible CFLP experiments; run from any directory.

python -B scripts/run_heuristic_experiments.py --iterations 2000

Exactly seeds 0..9 are attempted for each instance, ordered by numeric size.
Defaults match heuristic.py: T0=max(1, 0.01*Greedy cost), cooling=0.995.
Greedy is timed afresh per seed; SA timing excludes Greedy. Checkpoints after
each run retain completed results on interruption. Rerunning replaces only
the three heuristic experiment outputs (there is no automatic resume).
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

if __package__:
    from .cflp_common import evaluate, load_instance
    from .heuristic import GreedyConstructionError, MOVES, greedy_initial_solution, simulated_annealing
else:
    from cflp_common import evaluate, load_instance
    from heuristic import GreedyConstructionError, MOVES, greedy_initial_solution, simulated_annealing


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(10))
RESULT_FIELDS = (
    "instance", "customers", "sites", "seed", "status", "error", "feasible",
    "greedy_cost", "sa_cost", "absolute_improvement", "improvement_pct",
    "open_warehouses", "greedy_runtime_s", "sa_runtime_s", "total_runtime_s",
    "iterations", "initial_temperature", "cooling_rate", "instance_sha256",
)
HISTORY_FIELDS = ("instance", "seed", "iteration", "temperature", "current_cost", "best_cost")


def write_csv(path, fields, rows):
    """Replace one complete file atomically; never expose a truncated CSV."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _checkpoint(output_dir, metadata, runs, history):
    metadata["recorded_runs"] = len(runs)
    metadata["successful_runs"] = sum(r["status"] == "Completed" and r["feasible"] for r in runs)
    write_csv(output_dir / "heuristic_results.csv", RESULT_FIELDS,
              ({key: r[key] for key in RESULT_FIELDS} for r in runs))
    write_csv(output_dir / "sa_convergence.csv", HISTORY_FIELDS, history)
    path = output_dir / "heuristic_solutions.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "runs": runs},
                                    indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_experiments(instance_paths, output_dir, *, iterations=2000,
                    initial_temperature=None, cooling_rate=0.995):
    """Run ten seeds per instance and save results, assignments and history.

    Failed construction/search records are retained. An interrupted attempt is
    recorded and further runs stop. Input loading/configuration errors fail
    before replacing outputs. Timing excludes instance loading, final audit
    evaluation, and file I/O; total_runtime_s = initialization + search.
    """
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not math.isfinite(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if initial_temperature is not None and (
            not math.isfinite(initial_temperature) or initial_temperature <= 0):
        raise ValueError("initial_temperature must be finite and positive")
    loaded = [(Path(path), load_instance(path)) for path in instance_paths]
    loaded.sort(key=lambda pair: (pair[1]["n_customers"], pair[1]["n_facilities"],
                                  pair[1].get("name", pair[0].stem)))
    if not loaded:
        raise ValueError("No CFLP JSON instances found")
    names = [data.get("name", path.stem) for path, data in loaded]
    if len(set(names)) != len(names):
        raise ValueError("Instance names must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "seeds": list(SEEDS), "iterations": iterations,
        "initial_temperature_requested": initial_temperature,
        "default_temperature_rule": "max(1, 0.01 * greedy_cost)",
        "cooling_rate": cooling_rate, "neighborhoods": list(MOVES),
        "python_version": sys.version, "platform": platform.platform(),
        "expected_runs": len(loaded) * len(SEEDS), "all_runs_attempted": False,
        "interrupted": False, "instance_order": names,
        "timing_scope": "Greedy function and SA function timed separately; excludes input loading, final audit and output I/O",
        "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                          for name in ("cflp_common.py", "heuristic.py", "run_heuristic_experiments.py")},
    }
    runs, history = [], []
    _checkpoint(output_dir, metadata, runs, history)
    for path, data in loaded:
        name = data.get("name", path.stem)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in SEEDS:
            row = dict.fromkeys(RESULT_FIELDS)
            row.update(instance=name, customers=data["n_customers"], sites=data["n_facilities"],
                       seed=seed, status="GreedyFailed", error="", feasible=False,
                       iterations=iterations, cooling_rate=cooling_rate,
                       initial_temperature=initial_temperature, instance_sha256=digest,
                       greedy_runtime_s=0.0, sa_runtime_s=None, total_runtime_s=0.0,
                       initial_solution=None, solution=None)
            stage = "Greedy"
            started = time.perf_counter()
            try:
                initial = greedy_initial_solution(data)
                row["greedy_runtime_s"] = time.perf_counter() - started
                row["initial_solution"] = initial
                cost, feasible = evaluate(initial, data)
                if not feasible:
                    raise ValueError("Greedy returned an infeasible assignment")
                row["greedy_cost"] = cost
                row["initial_temperature"] = (max(1.0, 0.01 * cost)
                                              if initial_temperature is None else initial_temperature)
                stage = "SA"
                started = time.perf_counter()
                result = simulated_annealing(data, initial_solution=initial, seed=seed,
                                             iterations=iterations, initial_temperature=initial_temperature,
                                             cooling_rate=cooling_rate)
                row["sa_runtime_s"] = time.perf_counter() - started
                row["solution"] = result["solution"]
                final_cost, feasible = evaluate(result["solution"], data)
                row.update(sa_cost=final_cost, feasible=feasible,
                           open_warehouses=len(set(result["solution"])))
                if not feasible:
                    row.update(status="Infeasible", error="SA assignment exceeds capacity")
                elif not math.isclose(final_cost, result["objective"], rel_tol=1e-12, abs_tol=1e-8):
                    row.update(status="InvalidResult", error="Reported objective differs from evaluator")
                else:
                    row.update(status="Completed", absolute_improvement=cost-final_cost,
                               improvement_pct=100*(cost-final_cost)/cost if cost > 0 else None)
                history.extend({"instance": name, "seed": seed, **point} for point in result["history"])
            except KeyboardInterrupt:
                row.update(status="Interrupted", error=f"Interrupted during {stage}")
                metadata["interrupted"] = True
            except (GreedyConstructionError, ValueError, RuntimeError) as exc:
                row.update(status="GreedyFailed" if stage == "Greedy" else "SearchFailed", error=str(exc))
            finally:
                if stage == "Greedy" and row["greedy_runtime_s"] == 0:
                    row["greedy_runtime_s"] = time.perf_counter() - started
                elif stage == "SA" and row["sa_runtime_s"] is None:
                    row["sa_runtime_s"] = time.perf_counter() - started
                row["total_runtime_s"] = row["greedy_runtime_s"] + (row["sa_runtime_s"] or 0)
                runs.append(row)
                _checkpoint(output_dir, metadata, runs, history)
            print(f"{name} seed={seed}: {row['status']} cost={row['sa_cost']} "
                  f"time={row['total_runtime_s']:.3f}s", flush=True)
            if metadata["interrupted"]:
                return runs
    metadata["all_runs_attempted"] = True
    _checkpoint(output_dir, metadata, runs, history)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--initial-temperature", type=float, default=None)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    args = parser.parse_args()
    runs = run_experiments(args.instance_dir.glob("cflp_*.json"), args.output_dir,
                           iterations=args.iterations, initial_temperature=args.initial_temperature,
                           cooling_rate=args.cooling_rate)
    if any(row["status"] != "Completed" for row in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
