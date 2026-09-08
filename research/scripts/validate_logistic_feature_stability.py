"""Nested feature-set and regularization stability test for logistic regression."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_winprob.training.advanced import nested_logistic_feature_selection_benchmark
from nba_winprob.training.train import FEATURE_COLS

BASIC_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
]


def main() -> None:
    data = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    shooting = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *BASIC_SHOOTING_COLS],
    ).drop_duplicates("game_id")
    data = data.merge(shooting, on="game_id", how="left", validate="many_to_one")
    sets = {
        "base": FEATURE_COLS,
        "base_plus_basic_shooting": [*FEATURE_COLS, *BASIC_SHOOTING_COLS],
    }
    result = nested_logistic_feature_selection_benchmark(
        data,
        feature_sets=sets,
        min_train_seasons=2,
        c_grid=(0.05, 0.1, 0.5, 1.0, 3.0, 10.0),
    )
    output = Path("artifacts/logistic_feature_stability.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "metrics": result["metrics"],
        "outer_seasons": result["outer_seasons"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
