"""Tests for team context computation (rolling season stats)."""

import pytest

from nba_winprob.ingestion.team_context import build_season_context
from nba_winprob.schemas import TeamGameContext


def _row(game_id, game_date, team_id, matchup, wl, plus_minus):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": game_date,
        "TEAM_ID": team_id,
        "MATCHUP": matchup,
        "WL": wl,
        "PLUS_MINUS": plus_minus,
    }


TEAM_A = 1
TEAM_B = 2

# Three games: A beats B, then A loses to B, then A beats B again
SAMPLE_ROWS = [
    _row("g1", "2024-01-01", TEAM_A, "TMA vs. TMB", "W", 10),
    _row("g1", "2024-01-01", TEAM_B, "TMB @ TMA",   "L", -10),
    _row("g2", "2024-01-03", TEAM_A, "TMA @ TMB",   "L", -5),
    _row("g2", "2024-01-03", TEAM_B, "TMB vs. TMA", "W",  5),
    _row("g3", "2024-01-05", TEAM_A, "TMA vs. TMB", "W",  8),
    _row("g3", "2024-01-05", TEAM_B, "TMB @ TMA",   "L", -8),
]


class TestBuildSeasonContext:
    def test_returns_home_and_away_for_each_game(self):
        ctx = build_season_context(SAMPLE_ROWS)
        assert set(ctx["g1"]) == {"home", "away"}
        assert set(ctx["g2"]) == {"home", "away"}
        assert set(ctx["g3"]) == {"home", "away"}

    def test_season_opener_gets_neutral_prior(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # g1 is the first game for both teams — no prior data
        assert ctx["g1"]["home"].win_pct == 0.5
        assert ctx["g1"]["home"].avg_margin == 0.0
        assert ctx["g1"]["home"].streak == 0
        assert ctx["g1"]["away"].win_pct == 0.5

    def test_win_pct_reflects_prior_games_only(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # g2: team A has 1W-0L from g1, so win_pct = 1.0
        assert ctx["g2"]["away"].win_pct == pytest.approx(1.0)
        # g3: team A is 1W-1L from g1+g2, so win_pct = 0.5
        assert ctx["g3"]["home"].win_pct == pytest.approx(0.5)

    def test_avg_margin_reflects_prior_games(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # g2 away = team A, entering with margin of +10 from g1 → avg = 10.0
        assert ctx["g2"]["away"].avg_margin == pytest.approx(10.0)
        # g3 home = team A, entering with margins +10, -5 → avg = 2.5
        assert ctx["g3"]["home"].avg_margin == pytest.approx(2.5)

    def test_streak_win(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # Entering g2, team A just won g1 → streak = +1
        assert ctx["g2"]["away"].streak == 1

    def test_streak_loss_resets_win_streak(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # Entering g3, team A lost g2 after winning g1 → streak = -1
        assert ctx["g3"]["home"].streak == -1

    def test_venue_split_is_team_specific_and_time_aware(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # Entering g3, team A is 1-0 at home; team B is 0-1 on the road.
        assert ctx["g3"]["home"].venue_win_pct == pytest.approx(1.0)
        assert ctx["g3"]["away"].venue_win_pct == pytest.approx(0.0)

    def test_elo_is_prior_only_and_changes_after_results(self):
        ctx = build_season_context(SAMPLE_ROWS)
        assert ctx["g1"]["home"].elo_rating == pytest.approx(1500.0)
        assert ctx["g2"]["away"].elo_rating != pytest.approx(1500.0)

    def test_away_team_role_assigned_correctly(self):
        ctx = build_season_context(SAMPLE_ROWS)
        # g1: team B is the away team (matchup "TMB @ TMA")
        assert isinstance(ctx["g1"]["away"], TeamGameContext)

    def test_empty_input(self):
        assert build_season_context([]) == {}


class TestTeamContextIntegration:
    """FeatureVector carries context through GameState."""

    def test_gamestate_carries_context(self):
        from tests.conftest import make_event

        from nba_winprob.features import GameState

        ctx = TeamGameContext(win_pct=0.7, avg_margin=5.0, streak=3)
        state = GameState("g1", home_context=ctx)
        event = make_event(1, period=1, clock_seconds=700, home=0, away=0, game_id="g1")
        vec = state.update(event)

        assert vec.home_win_pct == pytest.approx(0.7)
        assert vec.home_avg_margin == pytest.approx(5.0)
        assert vec.home_streak == 3
        # Away defaults when no context given
        assert vec.away_win_pct == pytest.approx(0.5)
        assert vec.away_streak == 0

    def test_no_context_uses_neutral_defaults(self):
        from tests.conftest import make_event

        from nba_winprob.features import GameState

        state = GameState("g1")
        event = make_event(1, 1, 700, 0, 0, game_id="g1")
        vec = state.update(event)
        assert vec.home_win_pct == 0.5
        assert vec.away_win_pct == 0.5
        assert vec.home_streak == 0
