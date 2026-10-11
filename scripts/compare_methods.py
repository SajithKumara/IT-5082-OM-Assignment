"""Compare saved MILP and heuristic outputs without running either solver.

python -B scripts/compare_methods.py

Statistics use only Completed, feasible runs with finite nonnegative SA costs.
Std is the sample standard deviation (undefined with fewer than two runs).
MILP gaps require a finite positive cost and Feasible=True; only status Optimal
earns the label 'Optimality gap'. Blank MILP dual bounds remain unknown.
MILP timing is solve-only; heuristic timing includes initialization + search,
but neither includes instance loading or experiment output I/O.
Plotting requires matplotlib; CSV comparison uses only the standard library.
"""

import argparse
import csv
import math
from pathlib import Path
import statistics

if __package__:
    from .run_heuristic_experiments import write_csv
else:
    from run_heuristic_experiments import write_csv


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_FIELDS = (
    "instance", "customers", "sites", "recorded_runs", "expected_runs", "feasible_runs",
    "excluded_runs", "all_seeds_recorded", "milp_status", "milp_cost", "reference_valid",
    "reference_note", "gap_label", "best_sa_cost", "mean_sa_cost", "std_sa_cost",
    "best_gap_pct", "mean_gap_pct", "mean_greedy_cost", "milp_runtime_s",
    "mean_greedy_runtime_s", "mean_sa_runtime_s", "mean_total_runtime_s", "timing_note",
)
TIMING_NOTE = "MILP: solve only; heuristic: Greedy initialization + SA search; excludes experiment I/O"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def is_true(value):
    return value is True or str(value).strip().lower() == "true"


def valid_run(row):
    cost = number(row.get("sa_cost"))
    return row.get("status") == "Completed" and is_true(row.get("feasible")) and cost is not None and cost >= 0


def mean_field(rows, key):
    values = [number(row.get(key)) for row in rows]
    values = [v for v in values if v is not None and v >= 0]
    return statistics.mean(values) if values else None


def compare_results(milp_rows, heuristic_rows):
    """Match names explicitly; retain groups with no successful heuristic runs."""
    references = {}
    for row in milp_rows:
        name = row.get("Instance")
        if not name or name in references:
            raise ValueError("MILP instance names must be present and unique")
        references[name] = row
    groups = {}
    seen = set()
    for row in heuristic_rows:
        name, seed = row.get("instance"), number(row.get("seed"))
        if not name or seed is None or seed != int(seed) or not 0 <= seed <= 9:
            raise ValueError("Each heuristic run needs an instance name and seed 0..9")
        if (name, seed) in seen:
            raise ValueError(f"Duplicate heuristic instance/seed: {name}/{seed}")
        seen.add((name, seed))
        groups.setdefault(name, []).append(row)
    comparison = []
    for name, rows in groups.items():
        dimensions = {(int(r["customers"]), int(r["sites"])) for r in rows}
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent instance dimensions for {name}")
        n, m = dimensions.pop()
        reference = references.get(name, {})
        cost = number(reference.get("Cost (MILP)"))
        status = reference.get("Status", "").strip()
        ref_valid = (cost is not None and cost > 0 and is_true(reference.get("Feasible"))
                     and bool(status) and status.lower() not in ("infeasible", "unbounded", "undefined"))
        note = "Saved MILP reference; no solver rerun or dual bound assumed"
        if not reference:
            note = "Missing MILP reference"
        elif not ref_valid:
            note = "Invalid MILP reference: needs positive finite cost, feasible flag and usable status"
        elif number(reference.get("Customers")) != n or number(reference.get("Sites")) != m:
            ref_valid = False
            note = "MILP and heuristic instance dimensions differ"
        elif status.lower() != "optimal":
            note = "Feasible MILP reference is not proven optimal"
        good = [r for r in rows if valid_run(r)]
        costs = [number(r["sa_cost"]) for r in good]
        best, mean = (min(costs), statistics.mean(costs)) if costs else (None, None)
        if ref_valid and best is not None and best < cost - max(0.01, abs(cost)*1e-9):
            note += "; warning: SA cost is below saved MILP cost; investigate baseline"
        milp_runtime = number(reference.get("Runtime (s)"))
        comparison.append(dict(
            instance=name, customers=n, sites=m, recorded_runs=len(rows), expected_runs=10,
            feasible_runs=len(good), excluded_runs=len(rows)-len(good),
            all_seeds_recorded={int(number(r["seed"])) for r in rows} == set(range(10)),
            milp_status=status or None, milp_cost=cost if ref_valid else None,
            reference_valid=ref_valid, reference_note=note,
            gap_label=("Optimality gap (%)" if status.lower() == "optimal" else "Gap to MILP reference (%)") if ref_valid else "Unavailable",
            best_sa_cost=best, mean_sa_cost=mean,
            std_sa_cost=statistics.stdev(costs) if len(costs) >= 2 else None,
            best_gap_pct=100*(best-cost)/cost if ref_valid and best is not None else None,
            mean_gap_pct=100*(mean-cost)/cost if ref_valid and mean is not None else None,
            mean_greedy_cost=mean_field(good, "greedy_cost"),
            milp_runtime_s=milp_runtime if milp_runtime is not None and milp_runtime >= 0 else None,
            mean_greedy_runtime_s=mean_field(good, "greedy_runtime_s"),
            mean_sa_runtime_s=mean_field(good, "sa_runtime_s"),
            mean_total_runtime_s=mean_field(good, "total_runtime_s"), timing_note=TIMING_NOTE,
        ))
    return sorted(comparison, key=lambda r: (r["customers"], r["sites"], r["instance"]))


def print_comparison(rows):
    print("Instance             Runs       MILP cost       Best SA       Mean SA   Best gap%  Mean gap%  MILP s  Greedy s     SA s  Total s")
    for row in rows:
        def fmt(key, width, precision=2):
            value = row[key]
            return f"{value:{width},.{precision}f}" if value is not None else f"{'N/A':>{width}}"
        print(f"{row['instance']:<20} {row['feasible_runs']:>2}/{row['recorded_runs']:<2} "
              + " ".join(fmt(key, width, precision) for key, width, precision in (
                  ("milp_cost", 14, 2), ("best_sa_cost", 13, 2), ("mean_sa_cost", 13, 2),
                  ("best_gap_pct", 10, 3), ("mean_gap_pct", 10, 3), ("milp_runtime_s", 7, 2),
                  ("mean_greedy_runtime_s", 9, 4), ("mean_sa_runtime_s", 8, 3), ("mean_total_runtime_s", 8, 3))))
        if row["gap_label"] != "Optimality gap (%)" or not row["all_seeds_recorded"]:
            print(f"  {row['gap_label']}; {row['reference_note']}; all ten seeds recorded={row['all_seeds_recorded']}")
    print(TIMING_NOTE + ". Runtime means use successful feasible runs; std uses sample standard deviation.")


def create_plots(comparison, convergence_rows, output_dir, heuristic_rows):
    """Static export figures, numerically sorted; omit invalid costs/gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    output_dir = Path(output_dir)
    rows = sorted(comparison, key=lambda r: (r["customers"], r["sites"], r["instance"]))
    sizes = [r["customers"] for r in rows]
    def values(key):
        return [r[key] if r[key] is not None else float("nan") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(sizes, values("milp_cost"), "o-", label="Saved MILP reference")
    axes[0].plot(sizes, values("best_sa_cost"), "s-", label="Best feasible SA")
    axes[0].plot(sizes, values("mean_sa_cost"), "^-", label="Mean feasible SA")
    axes[0].set(title="Solution cost", xlabel="Customer cities", ylabel="Total opening + assignment cost")
    axes[1].plot(sizes, values("milp_runtime_s"), "o-", label="MILP solve only")
    axes[1].plot(sizes, values("mean_sa_runtime_s"), "s-", label="Mean SA search")
    axes[1].plot(sizes, values("mean_total_runtime_s"), "^-", label="Mean Greedy + SA")
    axes[1].set(title="Runtime (different timing scopes)", xlabel="Customer cities", ylabel="Seconds per run")
    positions = list(range(len(rows)))
    axes[2].bar([x-0.18 for x in positions], values("best_gap_pct"), width=0.36, label="Best SA")
    axes[2].bar([x+0.18 for x in positions], values("mean_gap_pct"), width=0.36, label="Mean SA")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xticks(positions, [r["instance"] for r in rows], rotation=25, ha="right")
    axes[2].set(title="SA gap to saved MILP reference", ylabel="100 × (SA − MILP) / MILP (%)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("CFLP: saved MILP versus Greedy + Simulated Annealing")
    fig.text(0.5, 0.015, "Gaps are optimality gaps only for valid Optimal MILP references. Runtime means exclude failed runs.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(output_dir / "heuristic_plots.png", dpi=160)
    plt.close(fig)

    valid_keys = {(r["instance"], int(r["seed"])) for r in heuristic_rows if valid_run(r)}
    grouped = {}
    for point in convergence_rows:
        key = point["instance"], int(point["seed"])
        if key in valid_keys:
            grouped.setdefault(key, []).append(point)
    columns = 2
    fig, axes = plt.subplots(max(1, (len(rows)+1)//2), columns,
                             figsize=(13, 4 * max(1, (len(rows)+1)//2)), squeeze=False)
    for ax, row in zip(axes.flat, rows):
        for seed in range(10):
            points = sorted(grouped.get((row["instance"], seed), []), key=lambda r: int(r["iteration"]))
            if points:
                ax.plot([int(r["iteration"]) for r in points],
                        [float(r["best_cost"]) for r in points], label=f"Seed {seed}", linewidth=1)
        ax.set(title=f"{row['instance']} — {row['feasible_runs']}/10 feasible runs",
               xlabel="SA iteration (0 = Greedy)", ylabel="Best total cost so far")
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(ncol=5, fontsize=7)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.suptitle("SA convergence: best-so-far cost for each recorded seed")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "heuristic_convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milp-csv", type=Path, default=ROOT / "data/milp_results.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = read_csv(args.data_dir / "heuristic_results.csv")
    comparison = compare_results(read_csv(args.milp_csv), runs)
    write_csv(args.data_dir / "heuristic_comparison.csv", COMPARISON_FIELDS, comparison)
    print_comparison(comparison)
    if not args.no_plots:
        create_plots(comparison, read_csv(args.data_dir / "sa_convergence.csv"), args.data_dir, runs)


if __name__ == "__main__":
    main()
