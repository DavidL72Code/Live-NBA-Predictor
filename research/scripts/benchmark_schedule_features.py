"""Walk-forward benchmark for historical schedule-derived features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_winprob.training.advanced import nested_game_xgb_benchmark
from nba_winprob.training.train import FEATURE_COLS

SCHEDULE_COLS = [
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_back_to_back", "away_back_to_back", "back_to_back_diff",
    "home_games_last_4_days", "away_games_last_4_days",
    "home_games_last_7_days", "away_games_last_7_days",
    "schedule_density_diff",
    "home_travel_miles", "away_travel_miles", "travel_miles_diff",
    "home_timezone_shift", "away_timezone_shift", "timezone_shift_diff",
]
PLAYER_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
    "home_rolling_top5_minutes_share", "away_rolling_top5_minutes_share",
    "home_rolling_rotation_size", "away_rolling_rotation_size",
    "home_lineup_continuity", "away_lineup_continuity",
]
FEATURE_GROUPS = {
    "schedule": SCHEDULE_COLS,
    "shooting": [
        "home_rolling_efg_pct", "away_rolling_efg_pct",
        "home_rolling_three_pct", "away_rolling_three_pct",
        "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
        "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
    ],
    "rotation_minutes": [
        "home_rolling_top5_minutes_share", "away_rolling_top5_minutes_share",
        "home_rolling_rotation_size", "away_rolling_rotation_size",
    ],
    "chemistry": ["home_lineup_continuity", "away_lineup_continuity"],
}
IMPROVED_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
    "home_rolling_fga", "away_rolling_fga",
    "home_rolling_three_pa", "away_rolling_three_pa",
    "home_rolling_fta", "away_rolling_fta",
    "home_shrunk_efg_pct", "away_shrunk_efg_pct",
    "home_shrunk_three_pct", "away_shrunk_three_pct",
    "home_shrunk_free_throw_pct", "away_shrunk_free_throw_pct",
    "shooting_diff_rolling_efg_pct", "shooting_diff_rolling_three_pct",
    "shooting_diff_rolling_free_throw_pct", "shooting_diff_rolling_true_shooting_pct",
    "shooting_diff_shrunk_efg_pct", "shooting_diff_shrunk_three_pct",
    "shooting_diff_shrunk_free_throw_pct",
]


def main() -> None:
    base = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    schedule = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *SCHEDULE_COLS, *PLAYER_COLS, *IMPROVED_SHOOTING_COLS],
    )
    schedule = schedule.drop_duplicates("game_id")
    data = base.merge(schedule, on="game_id", how="left", validate="many_to_one")
    all_extra = [*SCHEDULE_COLS, *PLAYER_COLS, *IMPROVED_SHOOTING_COLS]
    if data[all_extra].isna().any().any():
        raise ValueError("schedule/player feature merge has missing values")

    raw = nested_game_xgb_benchmark(data, n_estimators=100, n_jobs=2, feature_cols=FEATURE_COLS)
    isolated = {}
    for name, columns in FEATURE_GROUPS.items():
        isolated[name] = nested_game_xgb_benchmark(
            data,
            n_estimators=100,
            n_jobs=2,
            feature_cols=[*FEATURE_COLS, *columns],
        )
    improved_shooting = nested_game_xgb_benchmark(
        data,
        n_estimators=100,
        n_jobs=2,
        feature_cols=[*FEATURE_COLS, *IMPROVED_SHOOTING_COLS],
    )
    augmented = nested_game_xgb_benchmark(
        data,
        n_estimators=100,
        n_jobs=2,
        feature_cols=[*FEATURE_COLS, *SCHEDULE_COLS, *PLAYER_COLS],
    )
    result = {
        "dataset": {
            "rows": int(len(data)),
            "games": int(data.game_id.nunique()),
            "schedule_features": SCHEDULE_COLS,
            "player_team_features": PLAYER_COLS,
        },
        "raw_xgboost": raw,
        "isolated_groups": isolated,
        "improved_shooting": improved_shooting,
        "schedule_player_augmented_xgboost": augmented,
        "brier_delta_augmented_minus_raw": float(
            augmented["metrics"]["brier"] - raw["metrics"]["brier"]
        ),
        "logloss_delta_augmented_minus_raw": float(
            augmented["metrics"]["log_loss"] - raw["metrics"]["log_loss"]
        ),
    }
    output = Path("artifacts/live_schedule_features_benchmark.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "raw": raw["metrics"],
        "isolated": {
            name: {
                "metrics": result["metrics"],
                "brier_delta": result["metrics"]["brier"] - raw["metrics"]["brier"],
                "logloss_delta": result["metrics"]["log_loss"] - raw["metrics"]["log_loss"],
            }
            for name, result in isolated.items()
        },
        "improved_shooting": {
            "metrics": improved_shooting["metrics"],
            "brier_delta": improved_shooting["metrics"]["brier"] - raw["metrics"]["brier"],
            "logloss_delta": improved_shooting["metrics"]["log_loss"] - raw["metrics"]["log_loss"],
        },
        "augmented": augmented["metrics"],
        "brier_delta": result["brier_delta_augmented_minus_raw"],
        "logloss_delta": result["logloss_delta_augmented_minus_raw"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
