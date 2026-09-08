"""Benchmark enhanced lagged shooting features against raw XGBoost."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_winprob.training.advanced import nested_game_xgb_benchmark
from nba_winprob.training.train import FEATURE_COLS

NEW_SHOOTING_COLS = [
    "home_rolling_fga", "away_rolling_fga",
    "home_rolling_three_pa", "away_rolling_three_pa",
    "home_rolling_fta", "away_rolling_fta",
    "home_shrunk_efg_pct", "away_shrunk_efg_pct",
    "home_shrunk_three_pct", "away_shrunk_three_pct",
    "home_shrunk_free_throw_pct", "away_shrunk_free_throw_pct",
    "shooting_diff_rolling_efg_pct", "shooting_diff_rolling_three_pct",
    "shooting_diff_rolling_free_throw_pct",
    "shooting_diff_rolling_true_shooting_pct",
    "shooting_diff_shrunk_efg_pct", "shooting_diff_shrunk_three_pct",
    "shooting_diff_shrunk_free_throw_pct",
    "shooting_diff_rolling_fga", "shooting_diff_rolling_three_pa",
    "shooting_diff_rolling_fta",
]
BASE_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
]


def main() -> None:
    base = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    source = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *BASE_SHOOTING_COLS, *NEW_SHOOTING_COLS],
    ).drop_duplicates("game_id")
    data = base.merge(source, on="game_id", how="left", validate="many_to_one")
    all_features = [*BASE_SHOOTING_COLS, *NEW_SHOOTING_COLS]
    if data[all_features].isna().any().any():
        raise ValueError("shooting feature merge has missing values")

    raw = nested_game_xgb_benchmark(
        data, n_estimators=100, n_jobs=2, feature_cols=FEATURE_COLS
    )
    enhanced = nested_game_xgb_benchmark(
        data,
        n_estimators=100,
        n_jobs=2,
        feature_cols=[*FEATURE_COLS, *BASE_SHOOTING_COLS, *NEW_SHOOTING_COLS],
    )
    result = {
        "features": {"base_shooting": BASE_SHOOTING_COLS, "new": NEW_SHOOTING_COLS},
        "raw_xgboost": raw,
        "enhanced_shooting_xgboost": enhanced,
        "brier_delta_enhanced_minus_raw": (
            enhanced["metrics"]["brier"] - raw["metrics"]["brier"]
        ),
        "logloss_delta_enhanced_minus_raw": (
            enhanced["metrics"]["log_loss"] - raw["metrics"]["log_loss"]
        ),
    }
    output = Path("artifacts/live_improved_shooting_benchmark.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "raw": raw["metrics"],
        "enhanced": enhanced["metrics"],
        "brier_delta": result["brier_delta_enhanced_minus_raw"],
        "logloss_delta": result["logloss_delta_enhanced_minus_raw"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
