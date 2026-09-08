"""Benchmark only the basic lagged shooting feature group."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_winprob.training.advanced import nested_game_xgb_benchmark
from nba_winprob.training.train import FEATURE_COLS

BASIC_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
]


def main() -> None:
    base = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    source = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *BASIC_SHOOTING_COLS],
    ).drop_duplicates("game_id")
    data = base.merge(source, on="game_id", how="left", validate="many_to_one")
    if data[BASIC_SHOOTING_COLS].isna().any().any():
        raise ValueError("basic shooting feature merge has missing values")

    raw = nested_game_xgb_benchmark(
        data, n_estimators=100, n_jobs=2, feature_cols=FEATURE_COLS
    )
    shooting = nested_game_xgb_benchmark(
        data,
        n_estimators=100,
        n_jobs=2,
        feature_cols=[*FEATURE_COLS, *BASIC_SHOOTING_COLS],
    )
    result = {
        "features": BASIC_SHOOTING_COLS,
        "raw_xgboost": raw,
        "basic_shooting_xgboost": shooting,
        "brier_delta_shooting_minus_raw": (
            shooting["metrics"]["brier"] - raw["metrics"]["brier"]
        ),
        "logloss_delta_shooting_minus_raw": (
            shooting["metrics"]["log_loss"] - raw["metrics"]["log_loss"]
        ),
    }
    output = Path("artifacts/live_basic_shooting_benchmark.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "raw": raw["metrics"],
        "basic_shooting": shooting["metrics"],
        "brier_delta": result["brier_delta_shooting_minus_raw"],
        "logloss_delta": result["logloss_delta_shooting_minus_raw"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
