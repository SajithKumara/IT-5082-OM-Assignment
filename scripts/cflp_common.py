"""Standard-library CFLP loading, validation and Member A-compatible evaluation.

Assignments are single-source: solution[j] is a facility *list index*.
assign_cost[i][j] already includes the whole demand of customer j.
The solver-required fields are validated; descriptive Olist metadata is optional.
"""

import json
import math
from pathlib import Path


def _nonnegative_number(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be a finite nonnegative number")


def validate_instance(instance):
    """Validate dimensions, positional IDs (if supplied), demands and costs.

    Return the original dictionary without changing it. Insufficient capacity
    is allowed here: a well-formed instance can still be infeasible.
    """
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary")
    for key in ("n_customers", "n_facilities"):
        value = instance.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    n, m = instance["n_customers"], instance["n_facilities"]
    for key, count, fields in (
        ("customers", n, ("demand",)),
        ("facilities", m, ("capacity", "fixed_cost")),
    ):
        records = instance.get(key)
        if not isinstance(records, list) or len(records) != count:
            raise ValueError(f"{key} must be a list of length {count}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{key}[{index}] must be an object")
            if "id" in record and (type(record["id"]) is not int
                                   or record["id"] != index):
                raise ValueError(f"{key}[{index}].id must equal its list index")
            for field in fields:
                _nonnegative_number(record.get(field), f"{key}[{index}].{field}")
    for key in ("assign_cost", "distance_km"):
        if key == "distance_km" and key not in instance:
            continue
        matrix = instance.get(key)
        if not isinstance(matrix, list) or len(matrix) != m:
            raise ValueError(f"{key} must have {m} facility rows")
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(f"{key}[{i}] must have {n} customer columns")
            for j, value in enumerate(row):
                _nonnegative_number(value, f"{key}[{i}][{j}]")
    return instance


def load_instance(path):
    """Load a CFLP JSON file and validate its solver-required structure."""
    with Path(path).open(encoding="utf-8") as handle:
        return validate_instance(json.load(handle))


def evaluate(solution, instance):
    """Return (total_cost, capacity_feasible), using Member A's equations.

    Malformed assignments raise ValueError. A structurally valid assignment
    exceeding capacity returns its actual cost and False, like Member A.
    Used facilities are the open set; unused facilities incur no fixed cost.
    No rounding is applied.
    """
    validate_instance(instance)
    n, m = instance["n_customers"], instance["n_facilities"]
    if not isinstance(solution, (list, tuple)) or len(solution) != n:
        raise ValueError(f"solution must contain exactly {n} facility indices")
    if any(type(i) is not int or not 0 <= i < m for i in solution):
        raise ValueError(f"facility indices must be integers in [0, {m})")

    facilities = instance["facilities"]
    opened = set(solution)
    total_cost = sum(facilities[i]["fixed_cost"] for i in opened)
    total_cost += sum(instance["assign_cost"][i][j]
                      for j, i in enumerate(solution))
    loads = [0] * m
    for j, i in enumerate(solution):
        loads[i] += instance["customers"][j]["demand"]
    return total_cost, all(loads[i] <= facilities[i]["capacity"] for i in opened)
