"""
Fulcra Context MCP Server

A Model Context Protocol server providing access to Fulcra personal health data
with local caching for fast queries and offline access.

Environment Variables:
    FULCRA_ENVIRONMENT: Set to "stdio" for CLI mode, otherwise HTTP server mode
    FULCRA_DB_ENABLED: Set to "true" to enable local health database (default: true)
    FULCRA_DB_PATH: Path to SQLite database (default: ~/.fulcra_health_db/fulcra_health.db)
    FULCRA_DB_RECENT_DATA_HOURS: Hours to consider as "recent" data (default: 2)
    FULCRA_CHUNK_SIZE_DAYS: Days per chunk for large syncs (default: 7)
    FULCRA_MAX_RETRIES: Max retry attempts (default: 3)
    FULCRA_RETRY_BASE_DELAY: Base delay for exponential backoff (default: 1.0)
"""

from .health_db import HealthDatabase, get_health_db, reset_health_db
from .smart_fetch import SmartFetcher, get_smart_fetcher, reset_smart_fetcher

__all__ = [
    "HealthDatabase",
    "get_health_db",
    "reset_health_db",
    "SmartFetcher",
    "get_smart_fetcher",
    "reset_smart_fetcher",
]
