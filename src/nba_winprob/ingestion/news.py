"""Verified NBA availability and transaction context for analyst prompts.

This is deliberately an analyst context source, not a model feature. The model
must be retrained and benchmarked before news is allowed to alter its output.
Only official NBA pages are used, and every item carries a source and cutoff.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

_INJURY_ARCHIVE = "https://official.nba.com/nba-injury-report-2025-26-season/"
_MOVEMENT_FEED = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


@dataclass(frozen=True)
class NewsItem:
    category: str
    team: str
    text: str
    published_at: str
    source: str
    url: str


def _request(url: str, timeout: float = 15) -> requests.Response:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "nba-winprob/0.1 (local analyst context)"},
    )
    response.raise_for_status()
    return response


def _injury_items(teams: tuple[str, ...], cutoff: datetime) -> list[NewsItem]:
    """Find the newest official injury report published before the cutoff."""
    try:
        html = _request(_INJURY_ARCHIVE).text
        links = re.findall(r'href=["\']([^"\']+Injury-Report_[^"\']+\.pdf)["\']', html, re.I)
        candidates: list[tuple[datetime, str]] = []
        for href in links:
            match = re.search(r"Injury-Report_(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})", href)
            if not match:
                continue
            published = datetime(*map(int, match.groups()), tzinfo=timezone.utc)
            if published <= cutoff:
                candidates.append((published, urljoin(_INJURY_ARCHIVE, href)))
        if not candidates:
            return []
        published, url = max(candidates)
        pdf = _request(url).content
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            source.write(pdf)
            source.flush()
            text = subprocess.run(
                ["pdftotext", "-layout", source.name, "-"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout
        items: list[NewsItem] = []
        for team in teams:
            full_name = _TEAM_NAMES.get(team, team)
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if full_name.lower() not in line.lower() and not re.search(rf"\b{re.escape(team)}\b", line):
                    continue
                excerpt = " ".join(part.strip() for part in lines[index:index + 5] if part.strip())
                items.append(NewsItem("injury", team, excerpt[:500], published.isoformat(), "NBA official injury report", url))
                break
        return items
    except Exception as exc:
        logger.warning("official injury context unavailable: %s", exc)
        return []


def _transaction_items(teams: tuple[str, ...], cutoff: datetime) -> list[NewsItem]:
    """Read official player-movement records from the prior 30 days."""
    try:
        payload = _request(_MOVEMENT_FEED).json()
        raw = payload.get("NBA_Player_Movement", payload) if isinstance(payload, dict) else payload
        records = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
        start = cutoff - timedelta(days=30)
        items: list[NewsItem] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            blob = json.dumps(record).lower()
            team = next((code for code, name in _TEAM_NAMES.items() if code.lower() in blob or name.lower() in blob), None)
            if team not in teams:
                continue
            date_value = next((record.get(key) for key in ("TRANSACTION_DATE", "date", "transactionDate", "dateTransacted") if record.get(key)), None)
            try:
                published = datetime.fromisoformat(str(date_value).replace("Z", "+00:00")).astimezone(timezone.utc)
            except (TypeError, ValueError):
                continue
            if not start <= published <= cutoff:
                continue
            description = str(record.get("DESCRIPTION") or record.get("description") or record.get("TRANSACTION_TYPE") or "Official transaction")
            player = str(record.get("PLAYER_NAME") or record.get("playerName") or "Player movement")
            items.append(NewsItem("transaction", team, f"{player}: {description}", published.isoformat(), "NBA player movement feed", _MOVEMENT_FEED))
        return items[:20]
    except Exception as exc:
        logger.warning("official transaction context unavailable: %s", exc)
        return []


def fetch_team_news(teams: tuple[str, ...], cutoff: datetime | None = None) -> list[NewsItem]:
    """Fetch official, time-bounded context for the two teams."""
    cutoff = cutoff or datetime.now(timezone.utc)
    return _injury_items(teams, cutoff) + _transaction_items(teams, cutoff)
