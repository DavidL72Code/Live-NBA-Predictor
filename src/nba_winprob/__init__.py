"""Live NBA win probability system.

Package layout mirrors the architecture in nba-win-probability-plan.md:

- ``schemas``    — canonical event model shared by every layer
- ``gametime``   — game clock math (period/clock -> elapsed/remaining seconds)
- ``features``   — feature computation, single source of truth for both the
                   streaming (online) and batch (offline) paths
- ``ingestion``  — nba_api client, raw play-by-play normalization, backfill
"""

__version__ = "0.1.0"
