"""Rate-limited wrapper around nba_api's stats endpoints.

stats.nba.com is unauthenticated and unofficial; per the project plan we are
deliberately polite: a minimum interval between requests and bounded retries
with exponential backoff. All nba_api imports are lazy so unit tests and
feature code never need network-capable dependencies loaded.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def derive_opening_lineup(actions: list[dict], home_team_id: str, away_team_id: str) -> dict[str, list[dict]]:
    """Derive the five players on court before each team's first substitution.

    PlayByPlayV3 does not expose a complete lineup object, but the opening
    period records player actions and each team's first substitution. The
    unique player IDs before that substitution are the tip-off lineup for
    normal games. This helper is pure so the inference can be regression-tested.
    """
    team_ids = {"home": str(home_team_id), "away": str(away_team_id)}
    first_sub: dict[str, int] = {}
    for action in sorted(actions, key=lambda item: int(item.get("actionNumber") or 0)):
        if int(action.get("period") or 0) != 1:
            continue
        if str(action.get("actionType") or "").lower() != "substitution":
            continue
        team_id = str(action.get("teamId") or "")
        side = next((name for name, value in team_ids.items() if value == team_id), None)
        if side and side not in first_sub:
            first_sub[side] = int(action.get("actionNumber") or 0)

    result: dict[str, list[dict]] = {"home": [], "away": []}
    seen: dict[str, set[str]] = {"home": set(), "away": set()}
    for action in sorted(actions, key=lambda item: int(item.get("actionNumber") or 0)):
        if int(action.get("period") or 0) != 1:
            continue
        action_number = int(action.get("actionNumber") or 0)
        team_id = str(action.get("teamId") or "")
        side = next((name for name, value in team_ids.items() if value == team_id), None)
        if not side or action_number >= first_sub.get(side, 10**9):
            continue
        player_id = str(action.get("personId") or "")
        player_name = str(action.get("playerName") or "").strip()
        if not player_id or player_id == "0" or player_id in seen[side] or not player_name:
            continue
        seen[side].add(player_id)
        result[side].append({
            "player_id": player_id,
            "name": player_name,
            "team": side,
        })
    return result

DEFAULT_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


class NBAStatsClient:
    def __init__(
        self,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = self.min_request_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _call(self, description: str, make_request):
        """Run one endpoint request with throttling and backoff."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                return make_request()
            except Exception as exc:  # nba_api raises requests errors and JSON errors
                last_error = exc
                backoff = 2**attempt
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %ds",
                    description,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"{description} failed after {self.max_retries} attempts"
        ) from last_error

    def get_league_game_log(
        self, season: str, season_type: str = "Regular Season"
    ) -> list[dict]:
        """All team-game rows for a season with the fields needed to build team context.

        Returns one dict per team-game (two rows per game: home team and away team) with
        keys: GAME_ID, GAME_DATE, TEAM_ID, MATCHUP, WL, PLUS_MINUS.
        """
        from nba_api.stats.endpoints import leaguegamelog

        def make_request():
            return leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                timeout=self.timeout,
            ).get_dict()

        payload = self._call(f"LeagueGameLog({season}, {season_type})", make_request)
        result_set = payload["resultSets"][0]
        headers = result_set["headers"]

        wanted = {"GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "WL", "PLUS_MINUS"}
        indices = {h: i for i, h in enumerate(headers) if h in wanted}

        return [
            {field: row[idx] for field, idx in indices.items()}
            for row in result_set["rowSet"]
        ]

    def get_season_game_ids(
        self, season: str, season_type: str = "Regular Season"
    ) -> list[str]:
        """All game IDs for a season (e.g. season='2023-24'), in date order.

        LeagueGameLog lists each game once per team; duplicates are removed
        preserving first appearance.
        """
        from nba_api.stats.endpoints import leaguegamelog

        def make_request():
            return leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                timeout=self.timeout,
            ).get_dict()

        payload = self._call(f"LeagueGameLog({season}, {season_type})", make_request)
        result_set = payload["resultSets"][0]
        headers = result_set["headers"]
        game_id_idx = headers.index("GAME_ID")
        seen: dict[str, None] = {}
        for row in result_set["rowSet"]:
            seen.setdefault(str(row[game_id_idx]), None)
        return list(seen)

    def get_play_by_play_raw(self, game_id: str) -> dict:
        """Raw PlayByPlayV3 payload for one game (normalize separately).

        V3, not V2: as of mid-2025 the V2 endpoint returns empty payloads.
        """
        from nba_api.stats.endpoints import playbyplayv3

        def make_request():
            return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=self.timeout).get_dict()

        return self._call(f"PlayByPlayV3({game_id})", make_request)

    def get_boxscore(self, game_id: str) -> dict:
        """Live boxscore for one game.

        Returns a dict with keys:
            home_team (str): team abbreviation
            away_team (str): team abbreviation
            home_team_id (str)
            away_team_id (str)
            players (list[dict]): one entry per player with keys
                name, player_id, jersey, position, starter,
                team ("home"|"away"), minutes (raw str), points,
                assists, rebounds, fouls, plus_minus

        Makes two API calls: BoxScoreSummaryV2 (home/away IDs) and
        BoxScoreTraditionalV3 (player stats). Both share the client
        rate limiter so no extra throttling is needed.
        """
        from nba_api.stats.endpoints import boxscoresummaryv2, boxscoretraditionalv3

        # ── 1. Home / away team IDs ────────────────────────────────────────
        def get_summary():
            return boxscoresummaryv2.BoxScoreSummaryV2(
                game_id=game_id, timeout=self.timeout
            ).get_dict()

        try:
            summary = self._call(f"BoxScoreSummaryV2({game_id})", get_summary)
        except Exception as exc:
            # A stale/empty summary must not prevent the traditional boxscore
            # from supplying player names and IDs.
            logger.warning("summary unavailable for %s: %s", game_id, exc)
            summary = {}
        home_team_id = None
        away_team_id = None
        gs_rs = next((rs for rs in summary.get("resultSets", []) if rs.get("name") == "GameSummary"), None)
        if gs_rs and gs_rs.get("rowSet"):
            headers = gs_rs.get("headers", [])
            row = gs_rs["rowSet"][0]
            idx = {h: i for i, h in enumerate(headers)}
            if idx.get("HOME_TEAM_ID") is not None:
                home_team_id = str(row[idx["HOME_TEAM_ID"]])
            if idx.get("VISITOR_TEAM_ID") is not None:
                away_team_id = str(row[idx["VISITOR_TEAM_ID"]])
        summary_team_abbr: dict[str, str] = {}
        for rs in summary.get("resultSets", []):
            if rs.get("name") != "LineScore":
                continue
            line_headers = rs.get("headers", [])
            lh = {h: i for i, h in enumerate(line_headers)}
            team_id_idx = lh.get("TEAM_ID")
            abbr_idx = lh.get("TEAM_ABBREVIATION", lh.get("TEAM_CITY_NAME"))
            if team_id_idx is None or abbr_idx is None:
                continue
            for line_row in rs.get("rowSet", []):
                summary_team_abbr[str(line_row[team_id_idx])] = str(line_row[abbr_idx])

        # ── 2. Player stats ────────────────────────────────────────────────
        def get_trad():
            return boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=self.timeout
            ).get_dict()

        trad = self._call(f"BoxScoreTraditionalV3({game_id})", get_trad)
        result_sets = trad.get("resultSets", [])
        player_rs = next((rs for rs in result_sets if rs["name"] == "PlayerStats"), None)
        team_rs = next((rs for rs in result_sets if rs.get("name") == "TeamStats"), None)
        if (home_team_id is None or away_team_id is None) and team_rs:
            team_headers = {h: i for i, h in enumerate(team_rs.get("headers", []))}
            team_id_idx = team_headers.get("TEAM_ID")
            team_rows = team_rs.get("rowSet", [])
            team_ids = [str(row[team_id_idx]) for row in team_rows if team_id_idx is not None and row[team_id_idx] is not None]
            if home_team_id is None and team_ids:
                home_team_id = team_ids[0]
            if away_team_id is None and len(team_ids) > 1:
                away_team_id = team_ids[1]
        if player_rs is None:
            return {
                "home_team": summary_team_abbr.get(home_team_id, home_team_id),
                "away_team": summary_team_abbr.get(away_team_id, away_team_id),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "players": [],
            }
        ph = {h: i for i, h in enumerate(player_rs["headers"])}
        if team_rs is None:
            team_rs = {"headers": [], "rowSet": []}
        th = {h: i for i, h in enumerate(team_rs["headers"])}

        # Build team abbreviation lookup: team_id → abbreviation
        team_abbr: dict[str, str] = {}
        for trow in team_rs["rowSet"]:
            if "TEAM_ID" not in th:
                continue
            tid = str(trow[th["TEAM_ID"]])
            abbr = str(trow[th.get("TEAM_ABBREVIATION", th.get("TEAM_TRICODE", 0))])
            team_abbr[tid] = abbr

        home_team = team_abbr.get(home_team_id, home_team_id)
        away_team = team_abbr.get(away_team_id, away_team_id)

        players = []
        for prow in player_rs["rowSet"]:
            if "TEAM_ID" not in ph or ph["TEAM_ID"] >= len(prow):
                continue
            tid = str(prow[ph["TEAM_ID"]])
            side = "home" if tid == home_team_id else "away"
            player_id_idx = ph.get("PERSON_ID", ph.get("PLAYER_ID"))
            starter_idx = ph.get("START_POSITION")
            jersey_idx = ph.get("JERSEY_NUM", ph.get("JERSEY_NUMBER"))
            players.append({
                "name": prow[ph["PLAYER_NAME"]],
                "player_id": str(prow[player_id_idx]) if player_id_idx is not None else None,
                "jersey": str(prow[jersey_idx]) if jersey_idx is not None and prow[jersey_idx] is not None else "",
                "position": str(prow[starter_idx]) if starter_idx is not None and prow[starter_idx] is not None else "",
                "starter": bool(starter_idx is not None and str(prow[starter_idx]).strip()),
                "team": side,
                "minutes": prow[ph["MIN"]],
                "points": prow[ph["PTS"]],
                "assists": prow[ph["AST"]],
                "rebounds": prow[ph["REB"]],
                "fouls": prow[ph["PF"]],
                "plus_minus": prow[ph["PLUS_MINUS"]],
            })

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "players": players,
        }

    def get_team_roster(self, team_id: str, season: str) -> list[dict]:
        """Team roster for a season, normalized for UI player cards."""
        from nba_api.stats.endpoints import commonteamroster

        def get_roster():
            return commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=self.timeout,
            ).get_dict()

        payload = self._call(f"CommonTeamRoster({team_id}, {season})", get_roster)
        result_sets = payload.get("resultSets", [])
        roster_rs = next((rs for rs in result_sets if rs.get("name") == "CommonTeamRoster"), None)
        if roster_rs is None and result_sets:
            roster_rs = result_sets[0]
        if roster_rs is None:
            return []
        headers = roster_rs.get("headers", [])
        idx = {h: i for i, h in enumerate(headers)}
        players = []
        for row in roster_rs.get("rowSet", []):
            player_id_idx = idx.get("PLAYER_ID", idx.get("PERSON_ID"))
            if player_id_idx is None:
                continue
            player_id = str(row[player_id_idx])
            jersey_idx = idx.get("NUM", idx.get("JERSEY"))
            position_idx = idx.get("POSITION")
            name_idx = idx.get("PLAYER", idx.get("PLAYER_NAME"))
            players.append({
                "name": str(row[name_idx]) if name_idx is not None else "Player",
                "player_id": player_id,
                "jersey": str(row[jersey_idx]) if jersey_idx is not None and row[jersey_idx] is not None else "",
                "position": str(row[position_idx]) if position_idx is not None and row[position_idx] is not None else "",
                "starter": False,
                "points": 0,
                "assists": 0,
                "rebounds": 0,
                "image_url": f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png",
            })
        return players

    def get_opening_lineup(self, game_id: str, home_team_id: str, away_team_id: str) -> dict[str, list[dict]]:
        """Return players observed before each team's first Q1 substitution."""
        raw = self.get_play_by_play_raw(game_id)
        actions = raw.get("game", {}).get("actions", [])
        return derive_opening_lineup(actions, home_team_id, away_team_id)
