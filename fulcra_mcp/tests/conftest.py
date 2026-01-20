"""
Pytest configuration and shared fixtures for Fulcra Context MCP tests.

Provides fixtures for:
- Isolated test database
- Authenticated Fulcra API client
- Database cleanup
- Sample test data
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator

import httpx
import pytest
import structlog

from fulcra_mcp.health_db import HealthDatabase, get_health_db, reset_health_db
from fulcra_mcp.smart_fetch import SmartFetcher, get_smart_fetcher, reset_smart_fetcher

logger = structlog.getLogger(__name__)


# Test database path (isolated from production)
TEST_DB_PATH = Path.home() / ".fulcra_health_db" / "test_fulcra_health.db"
TEST_EXPORTS_PATH = Path.home() / ".fulcra_health_db" / "test_exports"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_path() -> Path:
    """
    Provide path to isolated test database.

    Uses a separate database file to avoid polluting production data.
    """
    return TEST_DB_PATH


@pytest.fixture(scope="session")
def test_exports_path() -> Path:
    """Provide path for test export files."""
    return TEST_EXPORTS_PATH


@pytest.fixture(scope="session")
def test_db(test_db_path: Path) -> Generator[HealthDatabase, None, None]:
    """
    Create and provide an isolated test database for the entire test session.

    Automatically creates the database schema and cleans up after tests.
    """
    # Ensure test database directory exists
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing test database if present
    if test_db_path.exists():
        test_db_path.unlink()

    logger.info("Creating test database", path=str(test_db_path))

    # Create test database instance
    db = HealthDatabase(db_path=test_db_path, enabled=True)

    yield db

    # Cleanup: Remove test database after all tests
    logger.info("Cleaning up test database", path=str(test_db_path))
    if test_db_path.exists():
        test_db_path.unlink()

    # Remove exports directory if exists
    if TEST_EXPORTS_PATH.exists():
        shutil.rmtree(TEST_EXPORTS_PATH)


@pytest.fixture
def clean_db(test_db: HealthDatabase) -> Generator[HealthDatabase, None, None]:
    """
    Provide a clean database for each test.

    Clears all tables before the test runs to ensure isolation.
    """
    # Clear all data before test (no confirm parameter needed)
    test_db.clear()

    yield test_db

    # Note: We don't clear after the test - this allows inspection if test fails


@pytest.fixture(scope="session")
async def fulcra_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Provide an authenticated Fulcra API client for the entire test session.

    Reads OAuth credentials from environment variables:
    - FULCRA_API: API endpoint
    - Access token should be available via OAuth flow

    Raises:
        ValueError: If required environment variables are not set
    """
    api_url = os.environ.get("FULCRA_API")
    if not api_url:
        pytest.skip("FULCRA_API environment variable not set - skipping integration tests")

    # Note: In real usage, the main.py handles OAuth flow
    # For tests, we assume the user is already authenticated
    # You may need to manually authenticate first or use a service account

    logger.info("Creating Fulcra API client", api_url=api_url)

    async with httpx.AsyncClient(
        base_url=api_url,
        timeout=30.0,
        follow_redirects=True
    ) as client:
        yield client


@pytest.fixture
def smart_fetcher(test_db: HealthDatabase) -> Generator[SmartFetcher, None, None]:
    """
    Provide a SmartFetcher instance configured for testing.

    Uses the test database and requires a Fulcra API client.
    """
    # Reset global smart fetcher to use test database
    reset_smart_fetcher()
    reset_health_db()

    # Set test database as global
    import fulcra_mcp.health_db as health_db_module
    health_db_module._health_db = test_db

    # Create smart fetcher (will use global test database)
    fetcher = get_smart_fetcher()

    yield fetcher

    # Reset after test
    reset_smart_fetcher()
    reset_health_db()


@pytest.fixture
def sample_metrics_data():
    """
    Provide sample metric data mimicking Fulcra API response.

    Includes numpy arrays and various timestamp formats to test serialization.
    """
    import numpy as np
    from datetime import datetime, timezone

    return [
        {
            "start_date": "2026-01-18T12:00:00Z",
            "end_date": "2026-01-18T12:01:00Z",
            "value": np.float64(72.5),  # Numpy scalar
            "metadata": {"source": "apple_watch"}
        },
        {
            "start_date": "2026-01-18T12:01:00Z",
            "end_date": "2026-01-18T12:02:00Z",
            "value": np.float64(73.0),
            "metadata": {"source": "apple_watch"}
        },
        {
            "start_date": datetime(2026, 1, 18, 12, 2, 0, tzinfo=timezone.utc),  # datetime object
            "end_date": datetime(2026, 1, 18, 12, 3, 0, tzinfo=timezone.utc),
            "value": 74.2,  # regular float
            "metadata": {"source": "apple_watch"}
        }
    ]


@pytest.fixture
def sample_workout_data():
    """
    Provide sample workout data mimicking Fulcra API response.

    Includes nested JSON structures to test serialization.
    """
    return [
        {
            "id": "workout_123",
            "start_date": "2026-01-18T07:00:00Z",
            "end_date": "2026-01-18T08:00:00Z",
            "workout_activity_type": "running",
            "duration": 3600,
            "total_distance": 10000.0,
            "metrics": {
                "avg_heart_rate": 145,
                "max_heart_rate": 178,
                "calories": 650
            }
        }
    ]


@pytest.fixture
def sample_sleep_data():
    """
    Provide sample sleep cycle data mimicking Fulcra API response.

    Includes complex nested structures.
    """
    return [
        {
            "id": "sleep_456",
            "start_date": "2026-01-17T23:00:00Z",
            "end_date": "2026-01-18T07:00:00Z",
            "sleep_stages": [
                {"stage": "awake", "start": "2026-01-17T23:00:00Z", "end": "2026-01-17T23:15:00Z"},
                {"stage": "light", "start": "2026-01-17T23:15:00Z", "end": "2026-01-18T00:00:00Z"},
                {"stage": "deep", "start": "2026-01-18T00:00:00Z", "end": "2026-01-18T02:00:00Z"},
                {"stage": "rem", "start": "2026-01-18T02:00:00Z", "end": "2026-01-18T03:00:00Z"}
            ],
            "total_sleep_duration": 28800
        }
    ]


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests requiring API access")
