#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, stable_int


def empirical_probability(prices: np.ndarray, bid: float) -> float:
    return float(np.mean(prices <= bid))


def utility(prices: np.ndarray, value: float, bid: float) -> float:
    return max(value - bid, 0.0) * empirical_probability(prices, bid)


def clean_auction_level(raw: pd.DataFrame, category: str) -> pd.DataFrame:
    part = raw.loc[raw["item"] == category].copy()
    inconsistent = part.groupby("auctionid")["openbid"].nunique()
    bad_ids = inconsistent[inconsistent > 1].index
    part = part.loc[~part["auctionid"].isin(bad_ids)].copy()
    return (
        part.groupby("auctionid", as_index=False)
        .agg(
            item=("item", "first"),
            openbid=("openbid", "first"),
            final_price=("price", "first"),
            n_bids=("bid", "size"),
            n_bidders=("bidder", "nunique"),
            auction_type=("auction_type", "first"),
        )
        .sort_values("auctionid")
        .reset_index(drop=True)
    )


def value_grid(prices: np.ndarray, count: int, step: int) -> list[int]:
    low = max(step * 2, int(math.ceil(np.quantile(prices, 0.35) / step) * step))
    high = max(low + step, int(math.ceil(np.quantile(prices, 0.95) * 1.25 / step) * step))
    raw = np.linspace(low, high, count)
    return sorted({int(round(x / step) * step) for x in raw})


def independent_candidates(
    value: int,
    train_prices: np.ndarray,
    option_count: int,
    step: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    market_low = max(step, int(math.floor(np.quantile(train_prices, 0.02) / step) * step))
    market_high = min(value - step, int(math.ceil(np.quantile(train_prices, 0.98) / step) * step))
    grid = np.arange(market_low, market_high + step, step, dtype=int)
    if len(grid) < option_count:
        grid = np.arange(step, value, step, dtype=int)
    if len(grid) < option_count:
        raise ValueError(f"Not enough feasible bids for value={value}, k={option_count}")
    # Candidate selection never conditions on the optimal bid. A random contiguous
    # window half the time and an unstructured sample half the time diversify gaps.
    if seed % 2 == 0 and len(grid) >= option_count + 2:
        start = int(rng.integers(0, len(grid) - option_count + 1))
        chosen = grid[start : start + option_count]
    else:
        chosen = np.sort(rng.choice(grid, size=option_count, replace=False))
    return [float(x) for x in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--auction-csv", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = ensure_dirs(args.output_dir)
    cfg = config["data"]
    raw = pd.read_csv(args.auction_csv)
    required = {"auctionid", "bid", "bidder", "openbid", "price", "item", "auction_type"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Auction CSV is missing columns: {sorted(missing)}")

    scenario_rows: list[dict] = []
    split_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for category in cfg["categories"]:
        auctions = clean_auction_level(raw, category)
        if len(auctions) < cfg["min_auctions"]:
            raise ValueError(f"{category} has only {len(auctions)} clean auctions")
        rng = np.random.default_rng(stable_int(config["seed"], category, "split"))
        order = rng.permutation(len(auctions))
        train_n = int(round(len(auctions) * cfg["train_fraction"]))
        train_idx = set(order[:train_n].tolist())
        auctions["split"] = ["train" if i in train_idx else "test" for i in range(len(auctions))]
        split_rows.append(auctions)
        train_prices = auctions.loc[auctions["split"] == "train", "final_price"].to_numpy(float)
        test_prices = auctions.loc[auctions["split"] == "test", "final_price"].to_numpy(float)
        summaries.append(
            {
                "category": category,
                "clean_auctions": len(auctions),
                "train_auctions": len(train_prices),
                "test_auctions": len(test_prices),
                "train_mean_price": float(np.mean(train_prices)),
                "test_mean_price": float(np.mean(test_prices)),
            }
        )
        values = value_grid(train_prices, cfg["values_per_category"], cfg["bid_step"])
        for value in values:
            for option_count in cfg["option_sizes"]:
                for set_index in range(cfg["candidate_sets_per_value"]):
                    candidate_seed = stable_int(config["seed"], category, value, option_count, set_index, "candidates")
                    bids = independent_candidates(value, train_prices, option_count, cfg["bid_step"], candidate_seed)
                    candidates = []
                    for bid in bids:
                        train_p = empirical_probability(train_prices, bid)
                        test_p = empirical_probability(test_prices, bid)
                        candidates.append(
                            {
                                "bid": bid,
                                "train_p": train_p,
                                "train_u": max(value - bid, 0.0) * train_p,
                                "test_p": test_p,
                                "test_u": max(value - bid, 0.0) * test_p,
                            }
                        )
                    train_best_position = int(np.argmax([x["train_u"] for x in candidates]))
                    test_best_position = int(np.argmax([x["test_u"] for x in candidates]))
                    scenario_id = f"{stable_int(category, value, option_count, set_index, modulo=10**12):012d}"
                    scenario_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "category": category,
                            "value": float(value),
                            "option_count": option_count,
                            "candidate_set_index": set_index,
                            "candidate_seed": candidate_seed,
                            "train_best_sorted_position": train_best_position,
                            "test_best_sorted_position": test_best_position,
                            "candidates_json": json.dumps(candidates, separators=(",", ":")),
                        }
                    )

    scenarios = pd.DataFrame(scenario_rows).sort_values(["category", "value", "option_count", "candidate_set_index"])
    scenarios.to_csv(paths["data"] / "scenarios.csv", index=False)
    pd.concat(split_rows, ignore_index=True).to_csv(paths["data"] / "auction_splits.csv", index=False)
    pd.DataFrame(summaries).to_csv(paths["data"] / "dataset_summary.csv", index=False)
    metadata = {
        "source_csv": str(Path(args.auction_csv).resolve()),
        "scenario_count": len(scenarios),
        "categories": sorted(scenarios["category"].unique().tolist()),
        "values": int(scenarios.groupby("category")["value"].nunique().min()),
        "config": config,
    }
    (paths["data"] / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

