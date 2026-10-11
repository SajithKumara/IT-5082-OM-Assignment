"""Fast experiment contract tests; no large datasets, MILP solves or plots."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heuristic_experiments as experiments
from scripts.cflp_common import evaluate
from scripts.compare_methods import compare_results, read_csv


def fixture(name="small", demand=None):
    demand = [2, 1] if demand is None else demand
    return {"name": name, "n_customers": len(demand), "n_facilities": 2,
            "customers": [{"id": j, "demand": d} for j, d in enumerate(demand)],
            "facilities": [{"id": i, "capacity": 3, "fixed_cost": 2} for i in range(2)],
            "assign_cost": [[1] * len(demand), [2] * len(demand)]}


def heuristic_row(name="small", seed=0, cost=120, **kwargs):
    row = {"instance": name, "customers": 2, "sites": 2, "seed": seed,
           "status": "Completed", "feasible": True, "sa_cost": cost,
           "greedy_cost": 150, "greedy_runtime_s": 1,
           "sa_runtime_s": 3, "total_runtime_s": 4}
    row.update(kwargs)
    return row


def milp_row(name="small", cost=100, status="Optimal", **kwargs):
    row = {"Instance": name, "Customers": "2", "Sites": "2", "Status": status,
           "Cost (MILP)": cost, "Feasible": "True", "Runtime (s)": "2"}
    row.update(kwargs)
    return row


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def save_fixture(self, data):
        path = self.folder / (data["name"] + ".json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_quietly(self, paths, output, iterations=4):
        with redirect_stdout(io.StringIO()):
            return experiments.run_experiments(paths, output, iterations=iterations)

    def test_seeds_numeric_order_schemas_and_reproducibility(self):
        small, larger = fixture("z_small"), fixture("a_large", [1, 1, 1])
        paths = [self.save_fixture(larger), self.save_fixture(small)]
        first = self.run_quietly(paths, self.folder / "one")
        second = self.run_quietly(reversed(paths), self.folder / "two")
        expected = [(name, seed) for name in ("z_small", "a_large") for seed in range(10)]
        self.assertEqual([(r["instance"], r["seed"]) for r in first], expected)
        for a, b in zip(first, second):
            self.assertEqual({k: v for k, v in a.items() if not k.endswith("runtime_s")},
                             {k: v for k, v in b.items() if not k.endswith("runtime_s")})
            self.assertEqual(a["total_runtime_s"], a["greedy_runtime_s"] + a["sa_runtime_s"])
            data = small if a["instance"] == "z_small" else larger
            self.assertEqual(evaluate(a["solution"], data), (a["sa_cost"], True))
        output = self.folder / "one"
        records = read_csv(output / "heuristic_results.csv")
        self.assertEqual(tuple(records[0]), experiments.RESULT_FIELDS)
        payload = json.loads((output / "heuristic_solutions.json").read_text())
        self.assertEqual(payload["runs"], first)
        self.assertEqual(payload["metadata"]["seeds"], list(range(10)))
        self.assertEqual(payload["metadata"]["expected_runs"], 20)
        self.assertTrue(payload["metadata"]["all_runs_attempted"])
        self.assertFalse(payload["metadata"]["interrupted"])
        self.assertEqual(len(payload["metadata"]["source_sha256"]), 3)
        convergence = read_csv(output / "sa_convergence.csv")
        self.assertEqual(tuple(convergence[0]), experiments.HISTORY_FIELDS)
        self.assertEqual(len(convergence), 20 * 5)
        self.assertEqual(convergence, read_csv(self.folder / "two/sa_convergence.csv"))
        self.assertFalse(list(output.glob("*.tmp")))

    def test_failed_initialization_is_recorded_for_every_seed(self):
        path = self.save_fixture(fixture("impossible", [7]))
        runs = self.run_quietly([path], self.folder / "output")
        self.assertEqual(len(runs), 10)
        for row in runs:
            self.assertEqual(row["status"], "GreedyFailed")
            self.assertTrue(row["error"])
            self.assertFalse(row["feasible"])
            self.assertIsNone(row["solution"])
            self.assertIsNone(row["sa_cost"])
        self.assertEqual(read_csv(self.folder / "output/sa_convergence.csv"), [])
        summary = compare_results([], runs)[0]
        self.assertEqual(summary["feasible_runs"], 0)
        self.assertEqual(summary["excluded_runs"], 10)
        self.assertIsNone(summary["best_sa_cost"])

    def test_infeasible_solver_result_not_reported_as_success(self):
        path = self.save_fixture(fixture("capacity", [2, 2]))
        bad = {"solution": [0, 0], "objective": 4, "history": []}
        with patch.object(experiments, "simulated_annealing", return_value=bad):
            rows = self.run_quietly([path], self.folder / "output")
        self.assertTrue(all(r["status"] == "Infeasible" and not r["feasible"] for r in rows))
        self.assertEqual(compare_results([], rows)[0]["feasible_runs"], 0)

    def test_objective_mismatch_and_search_failure(self):
        path = self.save_fixture(fixture())
        bad = {"solution": [0, 0], "objective": 999, "history": []}
        with patch.object(experiments, "simulated_annealing", return_value=bad):
            rows = self.run_quietly([path], self.folder / "invalid")
        self.assertTrue(all(r["status"] == "InvalidResult" for r in rows))
        with patch.object(experiments, "simulated_annealing", side_effect=RuntimeError("failure")):
            rows = self.run_quietly([path], self.folder / "failure")
        self.assertTrue(all(r["status"] == "SearchFailed" for r in rows))

    def test_interrupt_retains_partial_outputs(self):
        path = self.save_fixture(fixture())
        original = experiments.simulated_annealing
        calls = 0
        def interrupt_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original(*args, **kwargs)
        with patch.object(experiments, "simulated_annealing", side_effect=interrupt_second):
            rows = self.run_quietly([path], self.folder / "partial")
        self.assertEqual([r["status"] for r in rows], ["Completed", "Interrupted"])
        data = json.loads((self.folder / "partial/heuristic_solutions.json").read_text())
        self.assertTrue(data["metadata"]["interrupted"])
        self.assertFalse(data["metadata"]["all_runs_attempted"])
        self.assertEqual(data["metadata"]["recorded_runs"], 2)
        self.assertEqual(len(read_csv(self.folder / "partial/heuristic_results.csv")), 2)

    def test_invalid_configuration_does_not_create_outputs(self):
        path = self.save_fixture(fixture())
        for kwargs in ({"iterations": -1}, {"cooling_rate": 1}, {"initial_temperature": 0}):
            with self.assertRaises(ValueError):
                experiments.run_experiments([path], self.folder / "no-output", **kwargs)
        self.assertFalse((self.folder / "no-output").exists())


class ComparisonTests(unittest.TestCase):
    def test_name_matching_gap_formulas_and_sample_std(self):
        refs = [milp_row("other", 200), milp_row("small", 100)]
        runs = [heuristic_row("small", 1, 140), heuristic_row("other", 0, 210),
                heuristic_row("small", 0, 120)]
        rows = {r["instance"]: r for r in compare_results(refs, runs)}
        small = rows["small"]
        self.assertEqual(small["best_sa_cost"], 120)
        self.assertEqual(small["mean_sa_cost"], 130)
        self.assertAlmostEqual(small["std_sa_cost"], 200 ** 0.5)
        self.assertEqual(small["best_gap_pct"], 20)
        self.assertEqual(small["mean_gap_pct"], 30)
        self.assertEqual(rows["other"]["best_gap_pct"], 5)
        self.assertEqual(small["gap_label"], "Optimality gap (%)")
        self.assertEqual(small["mean_total_runtime_s"], 4)
        self.assertIsNone(rows["other"]["std_sa_cost"])
        self.assertFalse(small["all_seeds_recorded"])

    def test_nonoptimal_reference_is_not_an_optimality_gap(self):
        row = compare_results([milp_row(status="Not Solved")], [heuristic_row(cost=90)])[0]
        self.assertEqual(row["best_gap_pct"], -10)
        self.assertEqual(row["gap_label"], "Gap to MILP reference (%)")
        self.assertIn("not proven optimal", row["reference_note"])

    def test_missing_invalid_and_infeasible_references(self):
        refs = [[], [milp_row(cost="")], [milp_row(cost="nan")],
                [milp_row(cost="inf")], [milp_row(cost=0)], [milp_row(cost=-1)],
                [milp_row(Feasible="False")], [milp_row(status="Infeasible")],
                [milp_row(status="")], [milp_row(Customers="99")]]
        for reference in refs:
            with self.subTest(reference=reference):
                row = compare_results(reference, [heuristic_row()])[0]
                self.assertFalse(row["reference_valid"])
                self.assertIsNone(row["milp_cost"])
                self.assertIsNone(row["best_gap_pct"])
                self.assertEqual(row["gap_label"], "Unavailable")

    def test_exclude_failed_infeasible_and_invalid_costs(self):
        runs = [heuristic_row(seed=0, cost=120),
                heuristic_row(seed=1, cost=1, feasible="False"),
                heuristic_row(seed=2, cost=1, status="SearchFailed"),
                heuristic_row(seed=3, cost="NaN"), heuristic_row(seed=4, cost=-5)]
        row = compare_results([milp_row()], runs)[0]
        self.assertEqual(row["feasible_runs"], 1)
        self.assertEqual(row["excluded_runs"], 4)
        self.assertEqual(row["mean_sa_cost"], 120)

    def test_numeric_sort_and_duplicate_rejection(self):
        runs = [heuristic_row("large", customers=100), heuristic_row("small", customers=20)]
        self.assertEqual([r["customers"] for r in compare_results([], runs)], [20, 100])
        with self.assertRaises(ValueError):
            compare_results([], [heuristic_row(), heuristic_row()])
        with self.assertRaises(ValueError):
            compare_results([milp_row(), milp_row()], [])
        with self.assertRaises(ValueError):
            compare_results([], [heuristic_row(seed=10)])

    def test_csv_comparison_schema_and_no_invented_bounds(self):
        from scripts.compare_methods import COMPARISON_FIELDS
        rows = compare_results([milp_row()], [heuristic_row()])
        self.assertEqual(set(rows[0]), set(COMPARISON_FIELDS))
        self.assertFalse(any("bound" in key for key in rows[0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.csv"
            experiments.write_csv(path, COMPARISON_FIELDS, rows)
            loaded = read_csv(path)
            self.assertEqual(tuple(loaded[0]), COMPARISON_FIELDS)
            self.assertEqual(float(loaded[0]["best_gap_pct"]), 20)


if __name__ == "__main__":
    unittest.main()
