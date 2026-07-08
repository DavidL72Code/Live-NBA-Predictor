import pytest

from nba_winprob.ingestion.normalize import (
    SchemaDriftError,
    normalize_playbyplay,
    parse_clock,
    parse_score_value,
)
from nba_winprob.schemas import EventType


class TestParseClock:
    def test_v3_iso_format(self):
        assert parse_clock("PT12M00.00S") == 720
        assert parse_clock("PT11M43.00S") == 703
        assert parse_clock("PT00M02.50S") == 2.5

    def test_legacy_v2_format(self):
        assert parse_clock("12:00") == 720
        assert parse_clock("0:02.5") == 2.5

    def test_garbage_raises_drift_error(self):
        with pytest.raises(SchemaDriftError):
            parse_clock("eleven forty")


class TestParseScoreValue:
    def test_numeric_string(self):
        assert parse_score_value("121") == 121

    def test_blank_means_no_change(self):
        assert parse_score_value(None) is None
        assert parse_score_value("") is None
        assert parse_score_value("  ") is None

    def test_garbage_raises_drift_error(self):
        with pytest.raises(SchemaDriftError):
            parse_score_value("2 - 0")


class TestNormalize:
    def test_event_count_and_order(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)
        assert len(events) == 10
        assert [e.event_num for e in events] == list(range(1, 11))

    def test_actions_sorted_by_action_number(self, raw_pbp_payload):
        raw_pbp_payload["game"]["actions"].reverse()
        events = normalize_playbyplay(raw_pbp_payload)
        assert [e.event_num for e in events] == list(range(1, 11))

    def test_scores_forward_filled(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)
        # Event 4 (a missed shot, blank scores in the raw action) still carries 0-2
        miss = events[3]
        assert miss.event_type == EventType.FIELD_GOAL_MISSED
        assert (miss.home_score, miss.away_score) == (0, 2)
        # Final event has blank scores too but carries the final: home 5, away 4
        assert (events[-1].home_score, events[-1].away_score) == (5, 4)

    def test_event_types_mapped(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)
        assert events[0].event_type == EventType.PERIOD_START
        assert events[1].event_type == EventType.JUMP_BALL
        assert events[2].event_type == EventType.FIELD_GOAL_MADE
        assert events[-1].event_type == EventType.PERIOD_END

    def test_unknown_action_type_tolerated(self, raw_pbp_payload):
        raw_pbp_payload["game"]["actions"][1]["actionType"] = "Hologram Review"
        events = normalize_playbyplay(raw_pbp_payload)
        assert events[1].event_type == EventType.UNKNOWN

    def test_clock_and_descriptions(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)
        assert events[8].clock_seconds == 2.5
        assert "Home 3PT Shot" in events[8].description

    def test_missing_game_raises_drift_error(self, raw_pbp_payload):
        del raw_pbp_payload["game"]
        with pytest.raises(SchemaDriftError, match="game"):
            normalize_playbyplay(raw_pbp_payload)

    def test_missing_action_field_raises_drift_error(self, raw_pbp_payload):
        for action in raw_pbp_payload["game"]["actions"]:
            del action["scoreHome"]
        with pytest.raises(SchemaDriftError, match="scoreHome"):
            normalize_playbyplay(raw_pbp_payload)

    def test_empty_v2_style_payload_raises_drift_error(self):
        # What the dead PlayByPlayV2 endpoint returns as of mid-2025
        with pytest.raises(SchemaDriftError):
            normalize_playbyplay({})
