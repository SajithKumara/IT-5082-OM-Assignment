"""
Build Capacitated Facility Location (CFLP) instances from the Olist dataset.

Scenario: Olist wants to open regional warehouses. Each customer city must be
served by exactly one open warehouse (single-sourcing), warehouses have a
limited capacity, and we minimise fixed opening costs + transport costs.

Input  (download from Kaggle, place in data/raw/):
    olist_customers_dataset.csv
    olist_orders_dataset.csv
    olist_geolocation_dataset.csv

Output (data/processed/):
    city_demand.csv                 - all cities with coordinates and demand
    cflp_n{N}_m{M}.json             - one file per instance size

Run from the repo root:
    python scripts/preprocess.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Instance sizes: (number of customer cities, number of candidate warehouses)
# ---------------------------------------------------------------------------
INSTANCE_SIZES = [(20, 10), (50, 15), (100, 25), (200, 40)]

# ---------------------------------------------------------------------------
# Modelling assumptions (NOT in the data - justify these in the report)
# ---------------------------------------------------------------------------
COST_PER_ORDER_KM = 1.0    # transport cost of moving one order one km
FIXED_COST_RATIO = 1.0     # scales warehouse opening costs (see fixed_costs())
CAPACITY_SHARE = 0.30      # one warehouse can handle at most 30% of total demand

# Rough bounding box of Brazil, used to drop bad coordinates
LAT_MIN, LAT_MAX = -34.0, 5.5
LNG_MIN, LNG_MAX = -74.0, -34.0


def load_raw():
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    geo = pd.read_csv(RAW_DIR / "olist_geolocation_dataset.csv")
    return customers, orders, geo


def zip_coordinates(geo):
    """One (lat, lng) per zip prefix: median of all points, outliers removed."""
    geo = geo[
        geo["geolocation_lat"].between(LAT_MIN, LAT_MAX)
        & geo["geolocation_lng"].between(LNG_MIN, LNG_MAX)
    ]
    return (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def city_demand(customers, orders, zip_coords):
    """Demand = number of delivered orders per (city, state)."""
    delivered = orders[orders["order_status"] == "delivered"]
    df = delivered.merge(customers, on="customer_id", how="inner")
    df = df.merge(zip_coords, left_on="customer_zip_code_prefix",
                  right_index=True, how="inner")

    # City names can repeat across states (e.g. several "bom jesus"),
    # so a city is identified by (city, state).
    cities = (
        df.groupby(["customer_city", "customer_state"])
        .agg(demand=("order_id", "count"), lat=("lat", "median"), lng=("lng", "median"))
        .reset_index()
        .rename(columns={"customer_city": "city", "customer_state": "state"})
        .sort_values("demand", ascending=False, ignore_index=True)
    )
    return cities


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km. Accepts numpy arrays (broadcasts)."""
    r = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def fixed_costs(fac, assign_cost):
    """
    Opening cost per warehouse.

    Base level: the transport cost of serving everyone from the single best
    warehouse, spread over the candidate sites. This keeps fixed and transport
    costs on a similar scale, so the trade-off (open more warehouses vs. ship
    further) is meaningful.

    Larger cities get up to 25% higher costs (land/rent is more expensive),
    smaller ones up to 25% lower, based on their demand rank.
    """
    m = len(fac)
    single_best = assign_cost.sum(axis=1).min()
    base = FIXED_COST_RATIO * single_best / m
    size_rank = fac["demand"].rank(pct=True).to_numpy()  # 0..1, 1 = biggest
    return base * (0.75 + 0.5 * size_rank)


def build_instance(cities, n, m):
    cust = cities.head(n).reset_index(drop=True)
    fac = cust.head(m).reset_index(drop=True)   # candidate sites = biggest cities

    dist = haversine_km(fac["lat"].to_numpy()[:, None], fac["lng"].to_numpy()[:, None],
                        cust["lat"].to_numpy()[None, :], cust["lng"].to_numpy()[None, :])
    demand = cust["demand"].to_numpy()
    assign_cost = COST_PER_ORDER_KM * dist * demand[None, :]   # shape (m, n)

    total_demand = int(demand.sum())
    capacity = int(max(demand.max(), np.ceil(CAPACITY_SHARE * total_demand)))
    f = fixed_costs(fac, assign_cost)

    return {
        "name": f"cflp_n{n}_m{m}",
        "description": "Single-source CFLP built from Olist delivered orders",
        "n_customers": n,
        "n_facilities": m,
        "params": {
            "cost_per_order_km": COST_PER_ORDER_KM,
            "fixed_cost_ratio": FIXED_COST_RATIO,
            "capacity_share": CAPACITY_SHARE,
        },
        "customers": [
            {"id": j, "city": r.city, "state": r.state,
             "lat": round(r.lat, 5), "lng": round(r.lng, 5), "demand": int(r.demand)}
            for j, r in cust.iterrows()
        ],
        "facilities": [
            {"id": i, "city": r.city, "state": r.state,
             "lat": round(r.lat, 5), "lng": round(r.lng, 5),
             "fixed_cost": round(float(f[i]), 2), "capacity": capacity}
            for i, r in fac.iterrows()
        ],
        # assign_cost[i][j] = cost of serving customer j from facility i
        "assign_cost": np.round(assign_cost, 2).tolist(),
        "distance_km": np.round(dist, 2).tolist(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers, orders, geo = load_raw()
    cities = city_demand(customers, orders, zip_coordinates(geo))
    cities.to_csv(OUT_DIR / "city_demand.csv", index=False)
    print(f"{len(cities)} cities, {cities['demand'].sum()} delivered orders")

    for n, m in INSTANCE_SIZES:
        if n > len(cities):
            print(f"Skipping n={n}: only {len(cities)} cities available")
            continue
        inst = build_instance(cities, n, m)
        path = OUT_DIR / f"{inst['name']}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=1)
        cap = inst["facilities"][0]["capacity"]
        tot = sum(c["demand"] for c in inst["customers"])
        print(f"{path.name}: {n} customers, {m} sites, demand {tot}, "
              f"capacity/site {cap} (>= {int(np.ceil(tot / cap))} must open)")


if __name__ == "__main__":
    main()
