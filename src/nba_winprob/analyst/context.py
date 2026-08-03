"""Build rich game context for the LLM analyst.

``build_context`` fetches the live boxscore from NBA API and assembles a
``GameContext`` that bundles the model's feature vector, its win probability
estimate, and structured player stats. ``GameContext.to_prompt_text()`` renders
everything as plain text ready to paste into the analyst prompt.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nba_winprob.ingestion.client import NBAStatsClient
    from nba_winprob.schemas import FeatureVector
    from nba_winprob.ingestion.news import NewsItem

logger = logging.getLogger(__name__)

_TEAM_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def _resolve_team(abbr: str | None) -> str:
    if not abbr:
        return "Home"
    return _TEAM_NAMES.get(abbr.upper(), abbr)


def _team_code(value: str | None) -> str:
    """Normalize an abbreviation or full team name for official feeds."""
    if not value:
        return ""
    upper = value.upper()
    if upper in _TEAM_NAMES:
        return upper
    return next((code for code, name in _TEAM_NAMES.items() if name.upper() == upper), upper)


@dataclass
class PlayerStat:
    name: str
    team: str  # "home" or "away"
    minutes: float
    points: int
    assists: int
    rebounds: int
    fouls: int
    plus_minus: float


@dataclass
class GameContext:
    feature: "FeatureVector"
    model_prob: float
    home_team: str
    away_team: str
    players: list[PlayerStat] = field(default_factory=list)
    recent_plays: list[str] = field(default_factory=list)
    news_items: list["NewsItem"] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        fv = self.feature
        mins_left = fv.seconds_remaining / 60.0
        period_label = f"Q{fv.period}" if fv.period <= 4 else f"OT{fv.period - 4}"

        lines = [
            "GAME STATE",
            f"  HOME: {self.home_team} {fv.home_score}  vs  AWAY: {self.away_team} {fv.away_score}",
            f"  {period_label}, {mins_left:.1f} min remaining",
            f"  Model win probability (home): {self.model_prob:.1%}",
            "",
            "CONTEXT",
            f"  Score differential: {fv.score_diff:+d} (home perspective)",
            f"  Recent run (last 3 min): {self.home_team} +{fv.run_home}, {self.away_team} +{fv.run_away}",
            f"  Season W% — {self.home_team}: {fv.home_win_pct:.0%}  {self.away_team}: {fv.away_win_pct:.0%}",
            f"  Venue W% — {self.home_team} at home: {fv.home_venue_win_pct:.0%}  {self.away_team} on road: {fv.away_venue_win_pct:.0%}",
            f"  Venue margin — {self.home_team}: {fv.home_venue_avg_margin:+.1f}  {self.away_team}: {fv.away_venue_avg_margin:+.1f}",
            f"  Opponent-adjusted rating — {self.home_team}: {fv.home_elo_rating:.0f}  {self.away_team}: {fv.away_elo_rating:.0f}",
            f"  Streak — {self.home_team}: {_streak_str(fv.home_streak)}  {self.away_team}: {_streak_str(fv.away_streak)}",
        ]

        if self.players:
            home_players = sorted(
                [p for p in self.players if p.team == "home"], key=lambda p: -p.minutes
            )
            away_players = sorted(
                [p for p in self.players if p.team == "away"], key=lambda p: -p.minutes
            )
            lines += ["", "PLAYER STATS (sorted by minutes; ⚠ = 4+ fouls)"]
            lines.append(f"  {self.home_team}:")
            for p in home_players[:8]:
                flag = " ⚠" if p.fouls >= 4 else ""
                lines.append(
                    f"    {p.name}: {p.points}pts {p.assists}ast {p.rebounds}reb "
                    f"{p.fouls}pf{flag}  {p.plus_minus:+.0f}"
                )
            lines.append(f"  {self.away_team}:")
            for p in away_players[:8]:
                flag = " ⚠" if p.fouls >= 4 else ""
                lines.append(
                    f"    {p.name}: {p.points}pts {p.assists}ast {p.rebounds}reb "
                    f"{p.fouls}pf{flag}  {p.plus_minus:+.0f}"
                )

        if self.recent_plays:
            lines += ["", "RECENT PLAYS (most recent first)"]
            for play in self.recent_plays[:10]:
                lines.append(f"  • {play}")

        if self.news_items:
            lines += ["", "VERIFIED OFFICIAL CONTEXT (published before this game state)"]
            for item in self.news_items[:20]:
                lines.append(f"  [{item.category}] {item.team}: {item.text}")
                lines.append(f"    Source: {item.source} | {item.published_at} | {item.url}")
        else:
            lines += ["", "VERIFIED OFFICIAL CONTEXT", "  No official injury or transaction item was available before this game state."]

        return "\n".join(lines)


def _streak_str(streak: int) -> str:
    if streak > 0:
        return f"W{streak}"
    if streak < 0:
        return f"L{abs(streak)}"
    return "—"


def _parse_minutes(raw: str | None) -> float:
    """Parse NBA API minutes string to total minutes as a float."""
    if not raw:
        return 0.0
    s = str(raw).strip()
    # ISO 8601 duration: PT32M15.00S
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60.0
    # MM:SS
    m = re.match(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def build_context(
    game_id: str,
    feature: "FeatureVector",
    model_prob: float,
    client: "NBAStatsClient",
    recent_plays: list[str] | None = None,
    home_team_hint: str | None = None,
    away_team_hint: str | None = None,
    include_news: bool = False,
    news_cutoff: datetime | None = None,
    include_player_stats: bool = True,
) -> GameContext:
    """Fetch live boxscore and assemble a ``GameContext`` for the LLM analyst.

    Falls back to team name hints (or generic labels) if the boxscore call fails.
    """
    home_team = _resolve_team(home_team_hint) if home_team_hint else "Home"
    away_team = _resolve_team(away_team_hint) if away_team_hint else "Away"
    players: list[PlayerStat] = []
    news_items: list["NewsItem"] = []

    if include_player_stats:
        try:
            boxscore = client.get_boxscore(game_id)
            home_team = _resolve_team(boxscore.get("home_team")) or home_team
            away_team = _resolve_team(boxscore.get("away_team")) or away_team
            for p in boxscore.get("players", []):
                players.append(
                    PlayerStat(
                        name=p.get("name", "Unknown"),
                        team=p.get("team", "home"),
                        minutes=_parse_minutes(p.get("minutes")),
                        points=int(p.get("points") or 0),
                        assists=int(p.get("assists") or 0),
                        rebounds=int(p.get("rebounds") or 0),
                        fouls=int(p.get("fouls") or 0),
                        plus_minus=float(p.get("plus_minus") or 0),
                    )
                )
        except Exception as exc:
            logger.warning("boxscore fetch failed for %s: %s", game_id, exc)

    if include_news and home_team and away_team:
        from nba_winprob.ingestion.news import fetch_team_news

        news_items = fetch_team_news(
            (_team_code(home_team_hint or home_team), _team_code(away_team_hint or away_team)),
            news_cutoff,
        )

    return GameContext(
        feature=feature,
        model_prob=model_prob,
        home_team=home_team,
        away_team=away_team,
        players=players,
        recent_plays=recent_plays or [],
        news_items=news_items,
    )
