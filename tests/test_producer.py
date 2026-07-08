"""Producer unit tests — no broker, no network."""

import pytest

from nba_winprob.ingestion.normalize import SchemaDriftError
from nba_winprob.streaming.producer import LiveProducer


class TestPollNewEvents:
    def test_first_poll_returns_all_events(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}
        events = producer.poll_new_events("0022300001", raw_pbp_payload)
        assert len(events) == 10

    def test_cursor_advances_on_first_poll(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}
        events = producer.poll_new_events("0022300001", raw_pbp_payload)
        assert producer._cursors["0022300001"].last_action_num == events[-1].event_num

    def test_second_poll_with_no_new_events_returns_empty(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}
        producer.poll_new_events("0022300001", raw_pbp_payload)
        # same payload again — cursor already at the end
        second = producer.poll_new_events("0022300001", raw_pbp_payload)
        assert second == []

    def test_incremental_poll_returns_only_new(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}
        # Simulate first poll: only first 5 actions
        partial = dict(raw_pbp_payload)
        partial["game"] = dict(raw_pbp_payload["game"])
        partial["game"]["actions"] = raw_pbp_payload["game"]["actions"][:5]
        first = producer.poll_new_events("0022300001", partial)
        assert len(first) == 5
        # Full payload now includes all 10
        second = producer.poll_new_events("0022300001", raw_pbp_payload)
        assert len(second) == 5
        assert second[0].event_num == first[-1].event_num + 1

    def test_schema_drift_propagates(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}
        del raw_pbp_payload["game"]["actions"][0]["scoreHome"]
        with pytest.raises(SchemaDriftError):
            producer.poll_new_events("0022300001", raw_pbp_payload)

    def test_multiple_games_tracked_independently(self, raw_pbp_payload):
        producer = LiveProducer.__new__(LiveProducer)
        producer._cursors = {}

        payload_a = raw_pbp_payload
        payload_b = dict(raw_pbp_payload)
        payload_b["game"] = dict(raw_pbp_payload["game"], gameId="0022300002")
        for action in payload_b["game"]["actions"]:
            action = action  # actions share the list reference; we only check cursors
        # Deep copy actions for game B so they have a different game_id after normalization
        import copy
        payload_b["game"]["actions"] = copy.deepcopy(raw_pbp_payload["game"]["actions"])

        producer.poll_new_events("0022300001", payload_a)
        producer.poll_new_events("0022300002", payload_b)
        assert "0022300001" in producer._cursors
        assert "0022300002" in producer._cursors
        # Draining game A again returns nothing; B's cursor is separate
        assert producer.poll_new_events("0022300001", payload_a) == []
