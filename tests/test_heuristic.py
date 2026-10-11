"""Run from repo root: python -B -m unittest discover -s tests -v."""

import ast
import copy
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

from scripts.cflp_common import evaluate, load_instance, validate_instance
from scripts.heuristic import (
    GreedyConstructionError, MOVES, greedy_initial_solution,
    propose_neighbor, simulated_annealing,
)


ROOT = Path(__file__).resolve().parents[1]


def instance(demand, capacity, fixed, costs):
    return {
        "n_customers": len(demand), "n_facilities": len(capacity),
        "customers": [{"id": j, "demand": d} for j, d in enumerate(demand)],
        "facilities": [{"id": i, "capacity": c, "fixed_cost": fixed[i]}
                       for i, c in enumerate(capacity)],
        "assign_cost": costs,
    }


def exhaustive_solutions(data):
    """Independent reference: do not call the implementation's evaluator."""
    for assignment in itertools.product(range(data["n_facilities"]),
                                        repeat=data["n_customers"]):
        loads = [sum(c["demand"] for j, c in enumerate(data["customers"])
                     if assignment[j] == i) for i in range(data["n_facilities"])]
        if any(load > f["capacity"] for load, f in zip(loads, data["facilities"])):
            continue
        cost = 0
        for i, f in enumerate(data["facilities"]):
            if i in assignment:
                cost += f["fixed_cost"]
            for j in range(data["n_customers"]):
                if assignment[j] == i:
                    cost += data["assign_cost"][i][j]
        yield list(assignment), cost


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.data = instance([3, 2, 1], [5, 6], [10, 20],
                             [[1, 7, 9], [11, 2, 3]])

    def test_arithmetic_orientation_and_capacity(self):
        # Non-square matrix; demand must not multiply the stored costs again.
        self.assertEqual(evaluate([0, 0, 1], self.data), (41, True))
        self.assertEqual(evaluate([0, 0, 0], self.data), (27, False))
        self.assertEqual(evaluate([1, 1, 1], self.data), (36, True))

    def test_fixed_cost_once_and_no_rounding(self):
        data = instance([1, 1], [2], [10.12345], [[0.12345, 0.23456]])
        cost, feasible = evaluate([0, 0], data)
        self.assertTrue(feasible)
        self.assertAlmostEqual(cost, 10.48146)
        self.assertNotEqual(cost, round(cost, 2))

    def test_malformed_assignments(self):
        for solution in (None, "001", [], [0, 1], [0, 0, 1, 1],
                         [0, -1, 1], [0, 2, 1], [0, None, 1],
                         [0, 1.0, 1], [0, True, 1], [0, "1", 1]):
            with self.subTest(solution=solution), self.assertRaises(ValueError):
                evaluate(solution, self.data)

    def test_malformed_instances(self):
        bad = []
        for key in ("n_customers", "facilities", "assign_cost"):
            data = copy.deepcopy(self.data)
            del data[key]
            bad.append(data)
        for key, value in (("n_customers", True), ("n_facilities", 0),
                           ("assign_cost", [[1, 2], [3, 4], [5, 6]]),
                           ("distance_km", [[1]])):
            data = copy.deepcopy(self.data)
            data[key] = value
            bad.append(data)
        for section, field, value in (("customers", "demand", -1),
                                      ("customers", "id", 5),
                                      ("facilities", "capacity", float("inf")),
                                      ("facilities", "fixed_cost", float("nan")),
                                      ("facilities", "capacity", True)):
            data = copy.deepcopy(self.data)
            data[section][0][field] = value
            bad.append(data)
        data = copy.deepcopy(self.data)
        data["assign_cost"][0][0] = "bad"
        bad.append(data)
        for data in [None, []] + bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_instance(data)

    def test_load_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text(json.dumps(self.data), encoding="utf-8")
            self.assertEqual(load_instance(path), self.data)

    def test_exhaustive_reference(self):
        reference = dict((tuple(s), c) for s, c in exhaustive_solutions(self.data))
        for s in itertools.product(range(2), repeat=3):
            cost, feasible = evaluate(s, self.data)
            self.assertEqual(feasible, s in reference)
            if feasible:
                self.assertEqual(cost, reference[s])

    def test_member_a_evaluator_parity_without_running_notebook(self):
        notebook = json.loads((ROOT / "notebooks/milp_facility_location.ipynb")
                              .read_text(encoding="utf-8"))
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                definitions.extend(node for node in tree.body
                                   if isinstance(node, ast.FunctionDef)
                                   and node.name == "evaluate")
        self.assertEqual(len(definitions), 1)
        namespace = {}
        # Compile only the existing evaluator definition: no installs or solves.
        module = ast.Module(body=definitions, type_ignores=[])
        exec(compile(module, "member_a_evaluate", "exec"), namespace)
        for s in itertools.product(range(2), repeat=3):
            self.assertEqual(evaluate(s, self.data), namespace["evaluate"](s, self.data))


class GreedyTests(unittest.TestCase):
    def test_success_and_incremental_opening_cost(self):
        data = instance([3, 2], [5, 5], [10, 0], [[0, 0], [5, 5]])
        self.assertEqual(greedy_initial_solution(data), [1, 1])
        self.assertTrue(evaluate(greedy_initial_solution(data), data)[1])

    def test_demand_order_and_tie_breaking(self):
        data = instance([1, 4, 1], [4, 4], [0, 0], [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(greedy_initial_solution(data), [1, 0, 1])
        self.assertEqual(greedy_initial_solution(
            instance([2, 2], [2, 2], [0, 0], [[0, 0], [0, 0]])), [0, 1])

    def test_obvious_infeasibility(self):
        for data in (instance([6], [5, 5], [0, 0], [[0], [0]]),
                     instance([4, 4], [5], [0], [[0, 0]]),
                     instance([3, 3, 3], [5, 5], [0, 0], [[0]*3, [0]*3])):
            with self.assertRaises(GreedyConstructionError):
                greedy_initial_solution(data)

    def test_greedy_failure_does_not_mean_infeasible(self):
        data = instance([4, 3, 3], [6, 4], [0, 0], [[0, 0, 0], [1, 1, 1]])
        self.assertTrue(list(exhaustive_solutions(data)))
        with self.assertRaisesRegex(GreedyConstructionError, "not been proved"):
            greedy_initial_solution(data)


class NeighborhoodTests(unittest.TestCase):
    def test_all_moves_feasible_and_input_unchanged(self):
        data = instance([3, 2, 2, 1], [8, 8, 8, 8], [1, 1, 1, 1],
                        [[5]*4, [5]*4, [0]*4, [0]*4])
        original = [0, 0, 1, 1]
        saved_data = copy.deepcopy(data)
        reference = {tuple(s): cost for s, cost in exhaustive_solutions(data)}
        for move in MOVES:
            successes = 0
            rng = random.Random(42)
            for _ in range(40):
                candidate = propose_neighbor(original, data, rng, move)
                self.assertEqual(original, [0, 0, 1, 1])
                self.assertEqual(data, saved_data)
                if candidate is not None:
                    successes += 1
                    self.assertIn(tuple(candidate), reference)
                    self.assertEqual(evaluate(candidate, data),
                                     (reference[tuple(candidate)], True))
                    old_open, new_open = set(original), set(candidate)
                    if move == "open":
                        self.assertTrue(new_open - old_open)
                    if move == "close":
                        self.assertEqual(len(new_open), len(old_open) - 1)
                    if move == "facility_swap":
                        self.assertEqual(len(new_open), len(old_open))
                        self.assertEqual(len(new_open - old_open), 1)
            self.assertGreater(successes, 0, move)

    def test_impossible_moves_return_none(self):
        data = instance([4, 2], [4, 2], [0, 0], [[0, 0], [0, 0]])
        for move in MOVES:
            self.assertIsNone(propose_neighbor([0, 1], data, random.Random(0), move))
        with self.assertRaises(ValueError):
            propose_neighbor([0, 0], data, random.Random(0))
        with self.assertRaises(ValueError):
            propose_neighbor([0, 1], data, random.Random(0), "unknown")


class AnnealingTests(unittest.TestCase):
    def setUp(self):
        self.data = instance([2, 2, 1], [5, 5, 5], [4, 4, 4],
                             [[0, 6, 5], [6, 0, 5], [2, 2, 0]])

    def test_reproducibility_best_history_and_reference_bound(self):
        global_state = random.getstate()
        first = simulated_annealing(self.data, seed=42, iterations=200)
        second = simulated_annealing(self.data, seed=42, iterations=200)
        self.assertEqual(random.getstate(), global_state)
        for key in first:
            if key != "runtime_s":
                self.assertEqual(first[key], second[key], key)
        optimum = min(cost for _, cost in exhaustive_solutions(self.data))
        self.assertGreaterEqual(first["objective"], optimum)
        self.assertLessEqual(first["objective"], first["initial_cost"])
        self.assertEqual(evaluate(first["solution"], self.data),
                         (first["objective"], True))
        history = first["history"]
        self.assertEqual(len(history), 201)
        self.assertEqual([h["iteration"] for h in history], list(range(201)))
        self.assertTrue(all(b["best_cost"] <= a["best_cost"]
                            for a, b in zip(history, history[1:])))
        self.assertEqual(history[-1]["best_cost"], first["objective"])

    def test_zero_iterations_and_input_immutability(self):
        initial = [0, 1, 2]
        result = simulated_annealing(self.data, initial_solution=initial, iterations=0)
        self.assertEqual(result["solution"], initial)
        self.assertIsNot(result["solution"], initial)
        simulated_annealing(self.data, initial_solution=initial, iterations=50)
        self.assertEqual(initial, [0, 1, 2])

    def test_bad_parameters_and_initial_solution(self):
        for kwargs in ({"iterations": -1}, {"iterations": True}, {"seed": "42"},
                       {"cooling_rate": 1}, {"cooling_rate": 0},
                       {"cooling_rate": float("nan")}, {"initial_temperature": 0},
                       {"initial_temperature": float("inf")},
                       {"initial_solution": [None]*3}, {"initial_solution": "000"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulated_annealing(self.data, **kwargs)
        data = instance([2, 2], [2, 2], [0, 0], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            simulated_annealing(data, initial_solution=[0, 0])

    def test_tiny_temperature_is_safe(self):
        result = simulated_annealing(self.data, seed=1, iterations=100,
                                     initial_temperature=1e-320, cooling_rate=0.01)
        self.assertTrue(result["is_feasible"])


if __name__ == "__main__":
    unittest.main()
