"""Readable Greedy + feasible-state Simulated Annealing for single-source CFLP.

No external dependencies. Import with ``from scripts.heuristic import ...``
from the repository root, or import heuristic when scripts is on sys.path.
"""

import math
import random
import time

if __package__:
    from .cflp_common import evaluate, validate_instance
else:
    from cflp_common import evaluate, validate_instance


class GreedyConstructionError(ValueError):
    """Greedy got stuck; this is not a proof that the instance is infeasible."""


def greedy_initial_solution(instance):
    """Assign largest demands first, minimizing feasible incremental cost.

    Ties use customer/facility index. Opening cost is included only when the
    facility is first used. Failure is explicit; no partial result is returned.
    """
    validate_instance(instance)
    customers, facilities = instance["customers"], instance["facilities"]
    demand = [c["demand"] for c in customers]
    if sum(demand) > sum(f["capacity"] for f in facilities):
        raise GreedyConstructionError("Total demand exceeds total facility capacity")
    if any(d > max(f["capacity"] for f in facilities) for d in demand):
        raise GreedyConstructionError("A customer cannot fit in any facility")

    solution = [None] * len(customers)
    loads = [0] * len(facilities)
    opened = set()
    for j in sorted(range(len(customers)), key=lambda j: (-demand[j], j)):
        candidates = [i for i, f in enumerate(facilities)
                      if loads[i] + demand[j] <= f["capacity"]]
        if not candidates:
            raise GreedyConstructionError(
                f"Greedy cannot place customer {j}; construction failed, "
                "but instance infeasibility has not been proved"
            )
        i = min(candidates, key=lambda i: (
            instance["assign_cost"][i][j]
            + (0 if i in opened else facilities[i]["fixed_cost"]), i))
        solution[j] = i
        loads[i] += demand[j]
        opened.add(i)
    if not evaluate(solution, instance)[1]:
        raise GreedyConstructionError("Greedy construction failed its final capacity check")
    return solution


MOVES = ("reassign", "customer_swap", "open", "close", "facility_swap")


def propose_neighbor(solution, instance, rng, move=None):
    """Return a new feasible list, or None when one attempted move fails.

    Close/swap redistribute affected customers in descending demand order.
    They are bounded attempts, not exact assignment subproblems. Input is never
    modified. Opening a site always assigns a customer to it.
    """
    if not evaluate(solution, instance)[1]:
        raise ValueError("Neighborhoods require a feasible starting solution")
    if move is None:
        move = rng.choice(MOVES)
    if move not in MOVES:
        raise ValueError(f"Unknown neighborhood: {move}")
    n, m = instance["n_customers"], instance["n_facilities"]
    demand = [c["demand"] for c in instance["customers"]]
    facilities = instance["facilities"]
    opened = sorted(set(solution))
    closed = [i for i in range(m) if i not in opened]
    candidate = list(solution)

    if move == "reassign":
        j = rng.randrange(n)
        targets = [i for i in range(m) if i != solution[j]]
        if not targets:
            return None
        candidate[j] = rng.choice(targets)
    elif move == "customer_swap":
        if n < 2:
            return None
        j, k = rng.sample(range(n), 2)
        candidate[j], candidate[k] = candidate[k], candidate[j]
    elif move == "open":
        if not closed:
            return None
        target = rng.choice(closed)
        eligible = [j for j in range(n) if demand[j] <= facilities[target]["capacity"]]
        if not eligible:
            return None
        candidate[rng.choice(eligible)] = target
    else:
        if move == "close" and len(opened) < 2:
            return None
        if move == "facility_swap" and not closed:
            return None
        source = rng.choice(opened)
        target = rng.choice(closed) if move == "facility_swap" else None
        allowed = [i for i in opened if i != source]
        if target is not None:
            allowed.append(target)
        loads = [0] * m
        for j, i in enumerate(solution):
            if i != source:
                loads[i] += demand[j]
        affected = sorted((j for j in range(n) if solution[j] == source),
                          key=lambda j: (-demand[j], j))
        used = set(opened) - {source}
        for j in affected:
            fits = [i for i in allowed
                    if loads[i] + demand[j] <= facilities[i]["capacity"]]
            if not fits:
                return None
            i = min(fits, key=lambda i: (
                instance["assign_cost"][i][j]
                + (0 if i in used else facilities[i]["fixed_cost"]), i))
            candidate[j] = i
            loads[i] += demand[j]
            used.add(i)
        if target is not None and target not in candidate:
            return None

    if candidate == list(solution) or not evaluate(candidate, instance)[1]:
        return None
    return candidate


def simulated_annealing(instance, *, initial_solution=None, seed=0,
                        iterations=2000, initial_temperature=None,
                        cooling_rate=0.995):
    """Return the best feasible solution plus cost, timing and convergence.

    Default T0=max(1, 1% of initial cost); T <- max(T*cooling_rate, 1e-300).
    Each iteration attempts one uniformly selected neighborhood. Invalid moves
    are rejected and still consume an iteration/cooling step. Uphill moves use
    a log-domain Metropolis test to avoid exponential overflow/underflow.
    A fixed iteration budget and local RNG make assignments reproducible;
    elapsed runtime is naturally not reproducible. History includes iteration 0.
    runtime_s includes Greedy when no initial_solution is supplied.
    """
    validate_instance(instance)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if (isinstance(cooling_rate, bool) or not isinstance(cooling_rate, (int, float))
            or not math.isfinite(cooling_rate) or not 0 < cooling_rate < 1):
        raise ValueError("cooling_rate must be finite and between 0 and 1")
    if initial_temperature is not None and (
        isinstance(initial_temperature, bool)
        or not isinstance(initial_temperature, (int, float))
        or not math.isfinite(initial_temperature) or initial_temperature <= 0
    ):
        raise ValueError("initial_temperature must be finite and positive")
    started = time.perf_counter()
    current = (greedy_initial_solution(instance) if initial_solution is None
               else list(initial_solution) if isinstance(initial_solution, (list, tuple))
               else initial_solution)
    current_cost, feasible = evaluate(current, instance)
    if not feasible:
        raise ValueError("initial_solution exceeds capacity")
    current = list(current)
    best, best_cost = list(current), current_cost
    initial_cost = current_cost
    temperature = (max(1.0, 0.01 * initial_cost) if initial_temperature is None
                   else initial_temperature)
    starting_temperature = temperature
    rng = random.Random(seed)
    history = [{"iteration": 0, "temperature": temperature,
                "current_cost": current_cost, "best_cost": best_cost}]
    for iteration in range(1, iterations + 1):
        candidate = propose_neighbor(current, instance, rng)
        if candidate is not None:
            candidate_cost, feasible = evaluate(candidate, instance)
            if not feasible:
                raise RuntimeError("Neighborhood returned an infeasible assignment")
            delta = candidate_cost - current_cost
            accept = delta <= 0
            if not accept:
                uniform = rng.random()
                accept = uniform == 0 or math.log(uniform) < -delta / temperature
            if accept:
                current, current_cost = candidate, candidate_cost
                if current_cost < best_cost:
                    best, best_cost = list(current), current_cost
        history.append({"iteration": iteration, "temperature": temperature,
                        "current_cost": current_cost, "best_cost": best_cost})
        temperature = max(temperature * cooling_rate, 1e-300)
    return {
        "status": "Completed", "solution": best, "objective": best_cost,
        "initial_cost": initial_cost, "open_warehouses": sorted(set(best)),
        "is_feasible": evaluate(best, instance)[1],
        "runtime_s": time.perf_counter() - started,
        "seed": seed, "iterations": iterations,
        "initial_temperature": starting_temperature, "cooling_rate": cooling_rate,
        "history": history,
    }
