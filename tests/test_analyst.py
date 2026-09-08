"""Tests for the LLM analyst layer (serve, context, analyst)."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nba_winprob.analyst.analyst import AnalystOutput, LLMAnalyst
from nba_winprob.analyst.context import GameContext, PlayerStat, _parse_minutes, _streak_str, build_context
from nba_winprob.analyst.serve import LogisticWinProbServer, WinProbServer
from nba_winprob.schemas import FeatureVector
from nba_winprob.training.train import FEATURE_COLS


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_feature() -> FeatureVector:
    return FeatureVector(
        game_id="0022300001",
        event_num=150,
        period=4,
        seconds_remaining=120.0,
        seconds_elapsed=2160.0,
        home_score=98,
        away_score=95,
        score_diff=3,
        score_diff_norm=3 / (120.0 + 1.0) ** 0.5,
        run_home=8,
        run_away=2,
        run_diff=6,
        is_overtime=False,
        home_win_pct=0.62,
        home_avg_margin=4.5,
        home_streak=3,
        away_win_pct=0.55,
        away_avg_margin=2.1,
        away_streak=-1,
    )


@pytest.fixture()
def sample_context(sample_feature: FeatureVector) -> GameContext:
    players = [
        PlayerStat("LeBron James", "home", 34.5, 28, 7, 8, 4, +12),
        PlayerStat("Anthony Davis", "home", 32.0, 22, 2, 14, 2, +8),
        PlayerStat("Stephen Curry", "away", 35.0, 31, 6, 4, 3, -5),
        PlayerStat("Draymond Green", "away", 30.0, 4, 8, 9, 4, +2),
    ]
    return GameContext(
        feature=sample_feature,
        model_prob=0.67,
        home_team="LAL",
        away_team="GSW",
        players=players,
        recent_plays=["LeBron James makes 2-pt shot", "Stephen Curry misses 3-pt shot"],
    )


# ── _parse_minutes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PT32M15.00S", pytest.approx(32.25, abs=0.01)),
        ("32:15", pytest.approx(32.25, abs=0.01)),
        ("32.5", pytest.approx(32.5, abs=0.01)),
        ("PT0M0.00S", pytest.approx(0.0)),
        (None, 0.0),
        ("", 0.0),
        ("garbage", 0.0),
    ],
)
def test_parse_minutes(raw, expected):
    assert _parse_minutes(raw) == expected


# ── _streak_str ────────────────────────────────────────────────────────────────


def test_streak_str():
    assert _streak_str(3) == "W3"
    assert _streak_str(-2) == "L2"
    assert _streak_str(0) == "—"


# ── GameContext.to_prompt_text ─────────────────────────────────────────────────


def test_prompt_text_contains_key_info(sample_context: GameContext):
    text = sample_context.to_prompt_text()
    assert "LAL" in text
    assert "GSW" in text
    assert "67.0%" in text  # model_prob
    assert "LeBron James" in text
    assert "⚠" in text  # LeBron and Draymond have 4 fouls
    assert "RECENT PLAYS" in text


def test_prompt_text_no_players(sample_feature: FeatureVector):
    ctx = GameContext(feature=sample_feature, model_prob=0.5, home_team="BOS", away_team="MIA")
    text = ctx.to_prompt_text()
    assert "PLAYER STATS" not in text
    assert "BOS" in text


# ── WinProbServer ──────────────────────────────────────────────────────────────


def test_win_prob_server_predict(sample_feature: FeatureVector, tmp_path: Path):
    """WinProbServer.predict returns a float in [0, 1]."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    import xgboost as xgb

    # Tiny model trained on random data — we just check the pipeline runs
    rng = np.random.default_rng(0)
    X = rng.random((100, len(FEATURE_COLS)))
    y = rng.integers(0, 2, 100)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X, y)

    raw_probs = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y)

    cal_path = tmp_path / "cal.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    model_path = tmp_path / "model.ubj"
    model.save_model(str(model_path))

    server = WinProbServer.from_paths(model_path, cal_path)
    prob = server.predict(sample_feature)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_logistic_win_prob_server_predict(sample_feature: FeatureVector, tmp_path: Path):
    """The logistic shadow server loads and serves the shared feature contract."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    X = rng.random((100, len(FEATURE_COLS)))
    y = rng.integers(0, 2, 100)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000))
    model.fit(X, y)
    model_path = tmp_path / "logistic.pkl"
    with open(model_path, "wb") as artifact:
        pickle.dump(model, artifact)

    server = LogisticWinProbServer.from_paths(model_path)
    probability = server.predict(sample_feature)

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


# ── AnalystOutput ──────────────────────────────────────────────────────────────


def test_analyst_output_model():
    output = AnalystOutput(
        model_probability=0.67,
        analyst_probability=0.72,
        direction="higher",
        confidence="medium",
        headline="Lakers hold a commanding lead with Curry in foul trouble.",
        key_factors=["LeBron 28pts", "Curry 4 fouls"],
        reasoning="The model doesn't penalize individual foul trouble. With Curry at 4 fouls and 2 minutes left, Warriors will limit his minutes.",
    )
    assert output.direction == "higher"
    assert len(output.key_factors) == 2


# ── LLMAnalyst (mocked) ────────────────────────────────────────────────────────


def test_llm_analyst_analyze(sample_context: GameContext):
    """LLMAnalyst.analyze returns a valid AnalystOutput when the API responds correctly."""
    function_call = MagicMock()
    function_call.name = "submit_analysis"
    function_call.args = {
        "model_probability": 0.67,
        "analyst_probability": 0.72,
        "direction": "higher",
        "confidence": "medium",
        "headline": "Lakers in the driver's seat with Curry on the bench.",
        "key_factors": ["Curry foul trouble", "LeBron hot hand"],
        "reasoning": "The model underweights foul trouble. Curry sitting limits Warriors offense.",
    }

    part = MagicMock()
    part.function_call = function_call
    candidate = MagicMock()
    candidate.content.parts = [part]
    mock_response = MagicMock()
    mock_response.candidates = [candidate]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    analyst = LLMAnalyst.__new__(LLMAnalyst)
    analyst._client = mock_client
    analyst._model = "gemini-3.1-flash-lite"

    output = analyst.analyze(sample_context)

    assert isinstance(output, AnalystOutput)
    assert output.direction == "higher"
    assert output.analyst_probability == pytest.approx(0.72)
    assert len(output.key_factors) == 2


def test_llm_analyst_raises_if_no_tool_use(sample_context: GameContext):
    """LLMAnalyst.analyze raises RuntimeError when the model doesn't call the tool."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = []
    mock_response.candidates[0].finish_reason = "STOP"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    analyst = LLMAnalyst.__new__(LLMAnalyst)
    analyst._client = mock_client
    analyst._model = "gemini-3.1-flash-lite"

    with pytest.raises(RuntimeError, match="submit_analysis"):
        analyst.analyze(sample_context)


# ── build_context (mocked client) ─────────────────────────────────────────────


def test_build_context_success(sample_feature: FeatureVector):
    mock_client = MagicMock()
    mock_client.get_boxscore.return_value = {
        "home_team": "LAL",
        "away_team": "GSW",
        "home_team_id": "1610612747",
        "away_team_id": "1610612744",
        "players": [
            {
                "name": "LeBron James",
                "team": "home",
                "minutes": "PT34M30.00S",
                "points": 28,
                "assists": 7,
                "rebounds": 8,
                "fouls": 4,
                "plus_minus": 12.0,
            }
        ],
    }

    ctx = build_context("0022300001", sample_feature, 0.67, mock_client)

    assert ctx.home_team == "Los Angeles Lakers"
    assert ctx.away_team == "Golden State Warriors"
    assert len(ctx.players) == 1
    assert ctx.players[0].name == "LeBron James"
    assert ctx.players[0].fouls == 4


def test_build_context_api_failure(sample_feature: FeatureVector):
    """build_context falls back gracefully when the boxscore API call fails."""
    mock_client = MagicMock()
    mock_client.get_boxscore.side_effect = RuntimeError("network error")

    ctx = build_context("0022300001", sample_feature, 0.67, mock_client)

    assert ctx.home_team == "Home"
    assert ctx.away_team == "Away"
    assert ctx.players == []


def test_build_context_can_exclude_postgame_player_stats(sample_feature: FeatureVector):
    """Historical reads must not fetch final boxscore/player stats."""
    mock_client = MagicMock()

    ctx = build_context(
        "0022300001",
        sample_feature,
        0.67,
        mock_client,
        include_player_stats=False,
    )

    mock_client.get_boxscore.assert_not_called()
    assert ctx.players == []
