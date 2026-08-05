"""FastAPI server — NBA win-probability dashboard backend.

Endpoints
---------
GET  /                                   → serve the single-page UI
GET  /api/scoreboard?date=YYYY-MM-DD     → games for a date (defaults to today)
GET  /api/teams/{team_code}/games         → one team's season schedule
GET  /api/games/{game_id}/history        → full play-by-play + model probabilities
GET  /api/games/{game_id}/live           → current feature vector + probability (Redis)
GET  /api/games/{game_id}/stream         → SSE stream of live probability updates
POST /api/games/{game_id}/analyze        → trigger LLM analyst on live Redis state
POST /api/games/{game_id}/predict-at     → replay history to event_num, run model + LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from nba_winprob.config import get_settings

logger = logging.getLogger(__name__)

_UI_FILE = Path(__file__).resolve().parent.parent / "ui" / "index.html"
_RD_FILE = Path(__file__).resolve().parent.parent / "ui" / "rd.html"
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "ui" / "config.js"
_PLAYER_ROSTER_CACHE_FILE = Path(__file__).resolve().parent.parent / "ui" / "data" / "player_rosters_cache.json"
_PLAYER_ROSTER_CACHE_LOCK = threading.Lock()


def _allowed_origins() -> list[str]:
    settings = get_settings()
    return [
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ]


def _connect_sources() -> str:
    allowed = [origin for origin in _allowed_origins() if origin != "null"]
    return " ".join(["'self'", *allowed])


def _configure_stats_proxy() -> None:
    """Route nba_api through an optional same-schema cloud proxy."""
    proxy = (get_settings().stats_proxy_url or "").strip().rstrip("/")
    if not proxy:
        return
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.base_url = f"{proxy}/api/nba/stats/{{endpoint}}"
    token = (get_settings().stats_proxy_token or "").strip()
    if token:
        NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, "X-Swoosh-Proxy-Token": token}


app = FastAPI(title="NBA Win Probability Analyst", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach lightweight browser protections and a traceable request id."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path in {"/", "/rd.html"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https://cdn.nba.com https://fonts.googleapis.com https://fonts.gstatic.com; "
            "img-src 'self' data: https://cdn.nba.com https://a.espncdn.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            f"font-src 'self' https://fonts.gstatic.com; connect-src {_connect_sources()}; "
            "script-src 'self' 'unsafe-inline'"
        )
    return response

# ── Lazy model singleton ────────────────────────────────────────────────────

_prob_server = None
_server_loaded = False


class _RateLimiter:
    """Small in-process sliding-window limiter for quota-consuming requests."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window_seconds - now))
                return False, retry_after
            hits.append(now)
            return True, 0


# Both analyst endpoints call Gemini. Keep normal interactive use available
# while preventing a script or exposed dev server from burning the API quota.
_AI_PER_CLIENT_LIMITER = _RateLimiter(limit=6, window_seconds=60)
_AI_GLOBAL_LIMITER = _RateLimiter(limit=30, window_seconds=60)

_ANALYST_CACHE_TTL = 45.0
_ANALYST_CACHE: dict[str, tuple[float, dict]] = {}
_ANALYST_TASKS: dict[str, asyncio.Task] = {}
_ANALYST_CACHE_LOCK = asyncio.Lock()


def _feature_context(feature) -> dict:
    """Expose the actual model inputs so the UI can explain a prediction."""
    return {
        "score_diff": feature.score_diff,
        "run_home": feature.run_home,
        "run_away": feature.run_away,
        "home_win_pct": round(feature.home_win_pct, 4),
        "away_win_pct": round(feature.away_win_pct, 4),
        "home_venue_win_pct": round(feature.home_venue_win_pct, 4),
        "away_venue_win_pct": round(feature.away_venue_win_pct, 4),
        "home_venue_avg_margin": round(feature.home_venue_avg_margin, 2),
        "away_venue_avg_margin": round(feature.away_venue_avg_margin, 2),
        "home_elo_rating": round(feature.home_elo_rating, 1),
        "away_elo_rating": round(feature.away_elo_rating, 1),
        "home_streak": feature.home_streak,
        "away_streak": feature.away_streak,
    }


async def _cached_analyst(key: str, factory, ttl: float = _ANALYST_CACHE_TTL) -> dict:
    """Cache identical analyst states and deduplicate concurrent Gemini calls."""
    now = time.monotonic()
    async with _ANALYST_CACHE_LOCK:
        cached = _ANALYST_CACHE.get(key)
        if cached and now - cached[0] < ttl:
            return cached[1]
        task = _ANALYST_TASKS.get(key)
        if task is None:
            task = asyncio.create_task(factory())
            _ANALYST_TASKS[key] = task
    try:
        result = await task
        async with _ANALYST_CACHE_LOCK:
            _ANALYST_CACHE[key] = (time.monotonic(), result)
        return result
    finally:
        if task.done():
            async with _ANALYST_CACHE_LOCK:
                if _ANALYST_TASKS.get(key) is task:
                    _ANALYST_TASKS.pop(key, None)


def _check_ai_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    allowed, retry_after = _AI_PER_CLIENT_LIMITER.check(client_host)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="AI request limit reached for this client. Try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    allowed, retry_after = _AI_GLOBAL_LIMITER.check("all-clients")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="AI request capacity is temporarily exhausted. Try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )


@lru_cache(maxsize=4)
def _season_context(season: str) -> dict[str, dict]:
    """Load pregame team context once per season for historical serving."""
    _configure_stats_proxy()
    from nba_winprob.ingestion.client import NBAStatsClient
    from nba_winprob.ingestion.team_context import build_season_context

    rows = NBAStatsClient(min_request_interval=0.6, timeout=45).get_league_game_log(season)
    return build_season_context(rows)


def _game_season(game_id: str) -> str | None:
    """Derive the NBA season label from a standard game id (002YY...)."""
    if len(game_id) < 5 or not game_id[3:5].isdigit():
        return None
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _pregame_context(game_id: str) -> dict:
    season = _game_season(game_id)
    if not season:
        return {}
    try:
        return _season_context(season).get(game_id, {})
    except Exception as exc:
        logger.warning("could not load pregame context for %s: %s", game_id, exc)
        return {}


def _get_server():
    global _prob_server, _server_loaded
    if _server_loaded:
        return _prob_server
    _server_loaded = True
    settings = get_settings()
    if settings.analyst_mlflow_run_id:
        from nba_winprob.analyst.serve import WinProbServer

        try:
            _prob_server = WinProbServer.from_mlflow(
                settings.analyst_mlflow_run_id,
                tracking_uri=settings.mlflow_tracking_uri,
            )
            logger.info("model loaded from MLflow run %s", settings.analyst_mlflow_run_id)
        except Exception as exc:
            logger.warning("could not load model: %s — predictions will be unavailable", exc)
    return _prob_server


def _predict(feature) -> float | None:
    server = _get_server()
    return server.predict(feature) if server else None


def _predict_batch(features: list) -> list[float] | None:
    server = _get_server()
    return server.predict_batch(features) if server else None


# ── UI ──────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    if not _UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return _UI_FILE.read_text(encoding="utf-8")


@app.get("/rd.html", response_class=HTMLResponse, include_in_schema=False)
async def research_development():
    if not _RD_FILE.exists():
        raise HTTPException(status_code=404, detail="R&D page not found")
    return _RD_FILE.read_text(encoding="utf-8")


@app.get("/config.js", include_in_schema=False)
async def frontend_config():
    if _CONFIG_FILE.exists():
        return HTMLResponse(
            _CONFIG_FILE.read_text(encoding="utf-8"),
            media_type="application/javascript",
        )
    return HTMLResponse(
        "window.NBA_WINPROB_CONFIG = window.NBA_WINPROB_CONFIG || {};",
        media_type="application/javascript",
    )


# ── Scoreboard ──────────────────────────────────────────────────────────────


@app.get("/api/scoreboard")
async def get_scoreboard(date_param: str = Query(default=None, alias="date")):
    """Return all games for a date (YYYY-MM-DD) or today if omitted."""
    target_date = date_param or date.today().isoformat()
    try:
        games = await asyncio.to_thread(_fetch_scoreboard, target_date)
    except Exception as exc:
        logger.exception("scoreboard fetch failed for %s", target_date)
        detail = str(exc)
        if "stats.nba.com" in detail and ("Read timed out" in detail or "ConnectTimeout" in detail):
            detail = "NBA Stats did not respond from the hosting network before the 15-second limit."
        raise HTTPException(status_code=502, detail=detail) from exc
    return {"date": target_date, "games": games}


def _catalog_season() -> str:
    """Use the latest NBA season for the team catalog, even in the offseason."""
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


@app.get("/api/teams/{team_code}/games")
async def get_team_games(team_code: str, season: str | None = Query(default=None)):
    """Return a team's full latest-season schedule for catalog browsing."""
    code = team_code.strip().upper()
    valid_codes = {
        "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
        "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
    }
    if code not in valid_codes:
        raise HTTPException(status_code=400, detail="Unknown NBA team code")
    target_season = season or _catalog_season()
    try:
        games = await asyncio.to_thread(_fetch_team_schedule, code, target_season)
    except Exception as exc:
        logger.exception("team schedule fetch failed for %s %s", code, target_season)
        raise HTTPException(status_code=502, detail=str(exc))
    return {"team": code, "season": target_season, "games": games}


@lru_cache(maxsize=8)
def _fetch_team_schedule(team_code: str, season: str) -> list[dict]:
    import requests

    # NBA Stats' scheduleleaguev2 payload is frequently empty/malformed from
    # cloud hosts. ESPN's team-schedule endpoint currently ignores its season
    # query parameter, so build the requested season from monthly scoreboards.
    season_match = re.fullmatch(r"(\d{4})-(\d{2})", season)
    if not season_match:
        raise ValueError(f"invalid NBA season: {season}")
    start_year = int(season_match.group(1))
    end_year = start_year + 1
    month_specs = [(start_year, month) for month in range(9, 13)]
    month_specs.extend((end_year, month) for month in range(1, 7))

    last_error = ""
    events_by_id = {}
    for year, month in month_specs:
        month_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            month_days = 29
        date_range = f"{year:04d}{month:02d}01-{year:04d}{month:02d}{month_days:02d}"
        month_payload = None
        for hostname in ("site.web.api.espn.com", "site.api.espn.com"):
            try:
                response = requests.get(
                    f"https://{hostname}/apis/site/v2/sports/basketball/nba/scoreboard",
                    params={"dates": date_range, "limit": "1000"},
                    headers={"Accept": "application/json", "User-Agent": "SwooshAI/1.0"},
                    timeout=20,
                )
                if response.ok:
                    month_payload = response.json()
                    break
                last_error = f"{hostname} HTTP {response.status_code}: {response.text[:200]}"
            except (requests.RequestException, ValueError) as error:
                last_error = f"{hostname}: {error}"
        if month_payload is None:
            continue
        for event in month_payload.get("events", []):
            competitors = (event.get("competitions") or [{}])[0].get("competitors") or []
            if any((item.get("team") or {}).get("abbreviation") == team_code for item in competitors):
                events_by_id[str(event.get("id"))] = event
    if not events_by_id and last_error:
        raise RuntimeError(f"team schedule unavailable: {last_error}")

    games = []
    for event in events_by_id.values():
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), {})
        away = next((item for item in competitors if item.get("homeAway") == "away"), {})
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_code = str(home_team.get("abbreviation") or "")
        away_code = str(away_team.get("abbreviation") or "")
        if team_code not in {home_code, away_code}:
            continue

        status = competition.get("status") or {}
        status_type = status.get("type") or {}
        state = status_type.get("state") or "pre"
        if state == "in":
            status_label = "Live"
        elif state == "post":
            status_label = "Final"
        else:
            status_label = "Scheduled"
        season_type = event.get("seasonType") or {}
        type_id = int(season_type.get("type") or 2)
        label = str(event.get("name") or "").lower()
        if "cup" in label or "in-season" in label:
            game_type, game_type_label = "nba_cup", "NBA Cup"
        elif type_id == 1:
            game_type, game_type_label = "preseason", "Preseason"
        elif type_id == 3:
            game_type, game_type_label = "playoffs", "Playoffs"
        else:
            game_type, game_type_label = "regular", "Regular season"

        games.append({
            "game_id": f"espn:{event.get('id')}",
            "status": status_label,
            "status_text": str(status_type.get("detail") or status_type.get("shortDetail") or ""),
            "period": 0,
            "clock": "",
            "home_team": home_code,
            "home_team_name": str(
                home_team.get("shortDisplayName") or home_team.get("displayName") or ""
            ),
            "home_score": int(home.get("score") or 0),
            "away_team": away_code,
            "away_team_name": str(
                away_team.get("shortDisplayName") or away_team.get("displayName") or ""
            ),
            "away_score": int(away.get("score") or 0),
            "game_type": game_type,
            "game_type_label": game_type_label,
            "game_date": str(event.get("date") or ""),
            "game_time_utc": str(event.get("date") or ""),
        })
    return sorted(games, key=lambda game: game["game_date"])


def _fetch_scoreboard(target_date: str) -> list[dict]:
    settings = get_settings()
    proxy = (settings.stats_proxy_url or "").strip().rstrip("/")
    if proxy:
        import requests

        response = requests.get(
            f"{proxy}/api/nba/stats/scoreboardv3",
            params={"GameDate": target_date, "LeagueID": "00"},
            headers={"X-Swoosh-Proxy-Token": settings.stats_proxy_token or ""},
            timeout=15,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"NBA Stats proxy returned non-JSON HTTP {response.status_code}"
            ) from exc
        if response.status_code >= 400:
            detail = payload.get("detail") or payload.get("message") or str(payload)
            if response.status_code == 401 and "token_present" in payload:
                detail = f"{detail} (Render token present: {payload['token_present']})"
            raise RuntimeError(f"NBA Stats proxy HTTP {response.status_code}: {detail}")
    else:
        _configure_stats_proxy()
        from nba_api.stats.endpoints import scoreboardv3

        payload = scoreboardv3.ScoreboardV3(
            game_date=target_date,
            league_id="00",
            timeout=15,
        ).get_dict()
    if "scoreboard" not in payload:
        detail = (
            payload.get("detail")
            or payload.get("message")
            or payload.get("Message")
            or "NBA Stats returned an unexpected payload"
        )
        logger.warning(
            "unexpected scoreboard payload for %s; keys=%s", target_date, list(payload.keys())
        )
        raise RuntimeError(f"NBA Stats proxy/upstream error: {detail}")

    def clock_text(clock: str | None) -> str:
        value = str(clock or "").strip()
        match = re.fullmatch(r"PT(\d+)M([\d.]+)S", value)
        if not match:
            return value
        return f"{int(match.group(1)):02d}:{int(float(match.group(2))):02d}"

    def game_type(game: dict) -> tuple[str, str]:
        game_id = str(game.get("gameId", ""))
        label = " ".join(str(game.get(key, "") or "") for key in (
            "gameLabel", "gameSubLabel", "gameSubtype", "seriesText", "poRoundDesc"
        )).lower()
        if "cup" in label or "in-season" in label:
            return "nba_cup", "NBA Cup"
        if game_id.startswith("001"):
            return "preseason", "Preseason"
        if game_id.startswith("004") or game.get("seriesGameNumber") or game.get("poRoundDesc"):
            return "playoffs", "Playoffs"
        return "regular", "Regular season"

    games = []
    for game in payload.get("scoreboard", {}).get("games", []):
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        type_key, type_label = game_type(game)
        status_id = int(game.get("gameStatus") or 1)
        status_text = str(game.get("gameStatusText") or "").strip()
        status = {1: "Scheduled", 2: "Live", 3: "Final"}.get(status_id, status_text)
        games.append({
            "game_id": str(game.get("gameId", "")),
            "status": status,
            "status_text": status_text,
            "period": int(game.get("period") or 0),
            "clock": clock_text(game.get("gameClock")),
            "home_team": str(home.get("teamTricode") or ""),
            "home_team_name": str(home.get("teamName") or ""),
            "home_score": int(home.get("score") or 0),
            "away_team": str(away.get("teamTricode") or ""),
            "away_team_name": str(away.get("teamName") or ""),
            "away_score": int(away.get("score") or 0),
            "game_type": type_key,
            "game_type_label": type_label,
            "game_date": str(game.get("gameDate") or target_date),
            "game_time_utc": str(game.get("gameTimeUTC") or ""),
        })
    return games


# ── Game history ─────────────────────────────────────────────────────────────


@app.get("/api/games/{game_id}/history")
async def game_history(game_id: str, refresh: bool = Query(default=False)):
    """Full play-by-play for a game with model win probabilities at each event."""
    try:
        # Completed games are immutable, so reuse the in-process result.
        # Live games pass refresh=1 because their event stream must stay fresh.
        if refresh:
            result = await asyncio.to_thread(_fetch_history, game_id, True)
        else:
            result = await asyncio.to_thread(_fetch_history_cached, game_id)
    except Exception as exc:
        logger.exception("history fetch failed for %s", game_id)
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@app.get("/api/games/{game_id}/players")
async def game_players(game_id: str, refresh: bool = Query(default=False)):
    """Player cards for the roster-style pre-game/live intro."""
    try:
        if refresh:
            result = await asyncio.to_thread(_fetch_players, game_id)
        else:
            result = await asyncio.to_thread(_fetch_players_cached, game_id)
    except Exception as exc:
        logger.warning("player fetch failed for %s: %s", game_id, exc)
        return {
            "game_id": game_id,
            "home_team": None,
            "away_team": None,
            "home_team_id": None,
            "away_team_id": None,
            "home": [],
            "away": [],
            "available": False,
        }
    return result


@lru_cache(maxsize=32)
def _fetch_history_cached(game_id: str) -> dict:
    """Cache immutable completed-game timelines for fast repeat views."""
    return _fetch_history(game_id)


def _fetch_history(game_id: str, refresh: bool = False) -> dict:
    from nba_winprob.features.compute import GameState

    events = _load_normalized_events(game_id) if refresh else _normalized_events(game_id)

    if not events:
        return {"game_id": game_id, "events": []}

    context = _pregame_context(game_id)
    state = GameState(
        game_id,
        home_context=context.get("home"),
        away_context=context.get("away"),
    )
    pairs: list[tuple] = []
    for event in events:
        fv = state.update(event)
        pairs.append((event, fv))

    # Batch predict
    features = [fv for _, fv in pairs]
    probs = _predict_batch(features)

    result_events = []
    for i, (event, fv) in enumerate(pairs):
        result_events.append({
            "event_num": fv.event_num,
            "period": fv.period,
            "seconds_remaining": round(fv.seconds_remaining, 1),
            "seconds_elapsed": round(fv.seconds_elapsed, 1),
            "home_score": fv.home_score,
            "away_score": fv.away_score,
            "description": event.description or "",
            "event_type": int(event.event_type),
            "team_id": event.team_id,
            "team_tricode": event.team_tricode,
            "person_id": event.person_id,
            "player_name": event.player_name,
            "action_type": event.action_type,
            "sub_type": event.sub_type,
            "shot_result": event.shot_result,
            "shot_value": event.shot_value,
            "model_prob": round(probs[i], 4) if probs else None,
            "feature_context": _feature_context(fv),
        })

    return {
        "game_id": game_id,
        "events": result_events,
        "opening_context": _feature_context(pairs[0][1]),
        "latest_context": _feature_context(pairs[-1][1]),
    }


def _load_normalized_events(game_id: str) -> tuple:
    """Fetch and normalize one game's plays from its selected provider."""
    if game_id.startswith("espn:"):
        from nba_winprob.providers.espn import fetch_events

        return tuple(fetch_events(game_id))

    """Fetch and normalize one game's plays from NBA Stats."""
    _configure_stats_proxy()
    from nba_winprob.ingestion.client import NBAStatsClient
    from nba_winprob.ingestion.normalize import normalize_playbyplay

    client = NBAStatsClient(min_request_interval=0.6, timeout=45)
    raw = client.get_play_by_play_raw(game_id)
    return tuple(normalize_playbyplay(raw))


@lru_cache(maxsize=64)
def _fetch_players_cached(game_id: str) -> dict:
    return _fetch_players(game_id)


def _fetch_players(game_id: str) -> dict:
    if game_id.startswith("espn:"):
        from nba_winprob.providers.espn import fetch_players

        result = fetch_players(game_id)
        result["lineup_source"] = "espn_boxscore"
        _write_player_roster_cache(game_id, result)
        return result

    from nba_winprob.ingestion.client import NBAStatsClient

    _configure_stats_proxy()
    cached = _read_player_roster_cache().get(game_id)
    if cached and cached.get("lineup_source") in {"boxscore_start_position", "opening_play_by_play"}:
        return cached

    client = NBAStatsClient(min_request_interval=0.6, timeout=45)
    box = client.get_boxscore(game_id)
    players = box.get("players", [])
    lineup_source = "boxscore_start_position" if players and any(player.get("starter") for player in players) else None
    if players and not lineup_source:
        opening = client.get_opening_lineup(
            game_id, str(box.get("home_team_id")), str(box.get("away_team_id"))
        )
        players_by_id = {str(player.get("player_id")): player for player in players}
        opening_players = []
        for side in ("home", "away"):
            selected = [
                {**players_by_id[item["player_id"]], "team": side, "starter": True}
                for item in opening[side]
                if item["player_id"] in players_by_id
            ]
            opening_players.extend(selected if len(selected) == 5 else [])
        if len(opening_players) == 10:
            players = opening_players
            lineup_source = "opening_play_by_play"
    if not players:
        season = _season_from_game_id(game_id)
        home_roster = client.get_team_roster(str(box.get("home_team_id")), season)
        away_roster = client.get_team_roster(str(box.get("away_team_id")), season)
        roster_by_side = {"home": home_roster, "away": away_roster}
        opening = client.get_opening_lineup(
            game_id, str(box.get("home_team_id")), str(box.get("away_team_id"))
        )
        players = []
        for side, roster in roster_by_side.items():
            roster_by_id = {str(player.get("player_id")): player for player in roster}
            selected = [roster_by_id[item["player_id"]] for item in opening[side] if item["player_id"] in roster_by_id]
            if len(selected) == 5:
                players.extend({**player, "team": side, "starter": True} for player in selected)
            else:
                players.extend({**player, "team": side, "starter": False} for player in roster[:5])
        lineup_source = "opening_play_by_play" if len(players) == 10 and all(player.get("starter") for player in players) else "roster_preview"
    by_side = {"home": [], "away": []}

    def metric(player: dict) -> float:
        try:
            pts = float(player.get("points") or 0)
            reb = float(player.get("rebounds") or 0)
            ast = float(player.get("assists") or 0)
        except (TypeError, ValueError):
            return 0
        starter_bonus = 6 if player.get("starter") else 0
        return pts + reb * 0.7 + ast * 0.9 + starter_bonus

    for player in players:
        side = player.get("team")
        if side not in by_side:
            continue
        player_id = player.get("player_id")
        by_side[side].append({
            "name": player.get("name") or "Player",
            "player_id": player_id,
            "jersey": player.get("jersey") or "",
            "position": player.get("position") or "",
            "starter": bool(player.get("starter")),
            "points": player.get("points") or 0,
            "assists": player.get("assists") or 0,
            "rebounds": player.get("rebounds") or 0,
            "image_url": (
                player.get("image_url")
                or (
                    f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
                    if player_id
                    else None
                )
            ),
            "_metric": metric(player),
        })

    for side in by_side:
        by_side[side] = [
            {k: v for k, v in player.items() if k != "_metric"}
            for player in sorted(
                by_side[side],
                key=lambda p: (bool(p.get("starter")), p["_metric"]),
                reverse=True,
            )[:5]
        ]

    result = {
        "game_id": game_id,
        "home_team": box.get("home_team"),
        "away_team": box.get("away_team"),
        "home_team_id": box.get("home_team_id"),
        "away_team_id": box.get("away_team_id"),
        "home": by_side["home"],
        "away": by_side["away"],
        "lineup_source": lineup_source or "roster_preview",
    }
    _write_player_roster_cache(game_id, result)
    return result


def _read_player_roster_cache() -> dict:
    try:
        if not _PLAYER_ROSTER_CACHE_FILE.exists():
            return {}
        with _PLAYER_ROSTER_CACHE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("player roster cache read failed: %s", exc)
        return {}


def _write_player_roster_cache(game_id: str, payload: dict) -> None:
    try:
        with _PLAYER_ROSTER_CACHE_LOCK:
            _PLAYER_ROSTER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = _read_player_roster_cache()
            data[game_id] = payload
            tmp = _PLAYER_ROSTER_CACHE_FILE.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
            tmp.replace(_PLAYER_ROSTER_CACHE_FILE)
    except Exception as exc:
        logger.debug("player roster cache write failed for %s: %s", game_id, exc)


def _season_from_game_id(game_id: str) -> str:
    """Infer NBA season string from a stats.nba.com game id, e.g. 0022500688 → 2025-26."""
    try:
        start_two = int(str(game_id)[3:5])
    except (TypeError, ValueError):
        return f"{date.today().year}-{str(date.today().year + 1)[-2:]}"
    start_year = 2000 + start_two
    return f"{start_year}-{str(start_year + 1)[-2:]}"


@lru_cache(maxsize=64)
def _normalized_events(game_id: str) -> tuple:
    """Cache one immutable game's plays for repeat history/AI requests."""
    return _load_normalized_events(game_id)


# ── Live snapshot ─────────────────────────────────────────────────────────────


@app.get("/api/games/{game_id}/live")
async def game_live(game_id: str):
    """Current feature vector + model probability from Redis."""
    try:
        from nba_winprob.store.online import OnlineStore

        feature = OnlineStore().read(game_id)
    except Exception:
        feature = None

    if feature is None:
        return {"game_id": game_id, "available": False}

    prob = _predict(feature)
    return {
        "game_id": game_id,
        "available": True,
        "event_num": feature.event_num,
        "period": feature.period,
        "seconds_remaining": round(feature.seconds_remaining, 1),
        "seconds_elapsed": round(feature.seconds_elapsed, 1),
        "home_score": feature.home_score,
        "away_score": feature.away_score,
        "score_diff": feature.score_diff,
        "feature_context": _feature_context(feature),
        "model_prob": round(prob, 4) if prob is not None else None,
    }


# ── SSE live stream ───────────────────────────────────────────────────────────


@app.get("/api/games/{game_id}/stream")
async def stream_game(game_id: str):
    """Server-sent events: push probability updates as Redis changes."""

    async def generator():
        last_event_num = -1
        consecutive_misses = 0
        while True:
            try:
                from nba_winprob.store.online import OnlineStore

                feature = OnlineStore().read(game_id)
                if feature and feature.event_num != last_event_num:
                    last_event_num = feature.event_num
                    consecutive_misses = 0
                    prob = _predict(feature)
                    payload = {
                        "event_num": feature.event_num,
                        "period": feature.period,
                        "seconds_remaining": round(feature.seconds_remaining, 1),
                        "seconds_elapsed": round(feature.seconds_elapsed, 1),
                        "home_score": feature.home_score,
                        "away_score": feature.away_score,
                        "feature_context": _feature_context(feature),
                        "model_prob": round(prob, 4) if prob is not None else None,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    consecutive_misses += 1
                    # Heartbeat every 30s to keep connection alive
                    if consecutive_misses % 6 == 0:
                        yield ": heartbeat\n\n"
            except Exception as exc:
                logger.debug("stream error for %s: %s", game_id, exc)
                yield ": error\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── LLM analyst ──────────────────────────────────────────────────────────────


@app.post("/api/games/{game_id}/analyze", dependencies=[Depends(_check_ai_rate_limit)])
async def analyze_game(game_id: str):
    """Run the LLM analyst on the current game state."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=400, detail="NBA_WINPROB_GEMINI_API_KEY not configured"
        )

    try:
        from nba_winprob.store.online import OnlineStore

        feature = OnlineStore().read(game_id)
    except Exception:
        feature = None

    if feature is None:
        raise HTTPException(
            status_code=404,
            detail="No live feature vector for this game — is the streaming pipeline running?",
        )

    prob = _predict(feature)
    if prob is None:
        raise HTTPException(status_code=400, detail="Model not configured (set ANALYST_MLFLOW_RUN_ID)")

    from nba_winprob.analyst.analyst import LLMAnalyst
    from nba_winprob.analyst.context import build_context
    from nba_winprob.ingestion.client import NBAStatsClient

    async def generate():
        client = NBAStatsClient(min_request_interval=0.6)
        context = await asyncio.to_thread(
            build_context, game_id, feature, prob, client, include_news=True,
        )
        analyst = LLMAnalyst(api_key=settings.gemini_api_key, model=settings.analyst_model)
        output = await asyncio.to_thread(analyst.analyze, context)
        return {**output.model_dump(), "news_items": [item.__dict__ for item in context.news_items]}

    try:
        result = await _cached_analyst(f"live:{game_id}:{feature.event_num}", generate)
    except Exception as exc:
        logger.exception("analyst failed for %s", game_id)
        raise HTTPException(status_code=502, detail=str(exc))

    return {**result, "cached": True}


# ── Predict-at (historical replay) ───────────────────────────────────────────


class PredictAtRequest(BaseModel):
    event_num: int
    home_team: str | None = None
    away_team: str | None = None
    game_time_utc: str | None = None
    is_live: bool = False


def _replay_to_event(
    game_id: str, event_num: int, refresh: bool = False
) -> tuple:
    """Replay play-by-play up to event_num; return (FeatureVector, recent_plays)."""
    from nba_winprob.features.compute import GameState

    events = _load_normalized_events(game_id) if refresh else _normalized_events(game_id)

    context = _pregame_context(game_id)
    state = GameState(
        game_id,
        home_context=context.get("home"),
        away_context=context.get("away"),
    )
    target_fv = None
    desc_history: list[str] = []

    for event in events:
        if event.event_num > event_num:
            break
        fv = state.update(event)
        if event.description:
            desc_history.append(event.description)
        target_fv = fv  # last event at-or-before event_num

    if target_fv is None:
        raise ValueError(f"no events at or before event_num {event_num} in game {game_id}")

    # Most-recent plays first, capped at 15
    recent_plays = list(reversed(desc_history[-15:]))
    return target_fv, recent_plays


@app.post("/api/games/{game_id}/predict-at", dependencies=[Depends(_check_ai_rate_limit)])
async def predict_at_event(game_id: str, req: PredictAtRequest, refresh: bool = Query(default=False)):
    """Replay history to a specific event, run model + LLM analyst."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=400, detail="NBA_WINPROB_GEMINI_API_KEY not configured"
        )

    try:
        feature, recent_plays = await asyncio.to_thread(
            _replay_to_event, game_id, req.event_num, refresh
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("replay failed for %s event %d", game_id, req.event_num)
        raise HTTPException(status_code=502, detail=str(exc))

    prob = _predict(feature)
    if prob is None:
        raise HTTPException(
            status_code=400, detail="Model not configured (set NBA_WINPROB_ANALYST_MLFLOW_RUN_ID)"
        )

    from nba_winprob.analyst.analyst import LLMAnalyst
    from nba_winprob.analyst.context import build_context
    from nba_winprob.ingestion.client import NBAStatsClient

    async def generate():
        client = NBAStatsClient(min_request_interval=0.6)
        cutoff = None
        if req.game_time_utc:
            try:
                cutoff = datetime.fromisoformat(req.game_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                logger.warning("invalid game_time_utc for %s: %s", game_id, req.game_time_utc)
        context = await asyncio.to_thread(
            build_context, game_id, feature, prob, client, recent_plays,
            req.home_team, req.away_team,
            True, cutoff, req.is_live,
        )
        analyst = LLMAnalyst(api_key=settings.gemini_api_key, model=settings.analyst_model)
        output = await asyncio.to_thread(analyst.analyze, context)
        return {**output.model_dump(), "news_items": [item.__dict__ for item in context.news_items]}

    try:
        # Version the key so cached responses generated before the leak fix
        # cannot be reused for historical reads.
        result = await _cached_analyst(f"replay:v2:{game_id}:{feature.event_num}", generate, ttl=3600)
    except Exception as exc:
        logger.exception("analyst failed for %s event %d", game_id, req.event_num)
        raise HTTPException(status_code=502, detail=str(exc))

    period_label = f"Q{feature.period}" if feature.period <= 4 else f"OT{feature.period - 4}"
    return {
        **result,
        "event_num": req.event_num,
        "period_label": period_label,
        "seconds_remaining": round(feature.seconds_remaining, 1),
        "home_score": feature.home_score,
        "away_score": feature.away_score,
        "model_prob": round(prob, 4),
        "feature_context": _feature_context(feature),
        "cached": True,
    }
