"""
Health Database Module - Personal Health Data Warehouse

A local SQLite database that acts as a permanent health data warehouse.
Once data is fetched from Fulcra, it's stored forever for fast queries
and offline access.

Core Principles:
1. Immutable Data Philosophy: Health data doesn't change once recorded
2. Fetch Once, Store Forever: Minimize API calls
3. Smart Sync: Only fetch data that doesn't exist locally
4. Recent Data Exception: Very recent data (last 2 hours) may still be updating
"""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.getLogger(__name__)


@dataclass
class TimeGap:
    """Represents a gap in time-series data that needs fetching."""

    start_time: float  # Unix timestamp
    end_time: float  # Unix timestamp

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time) / 3600


@dataclass
class SyncProgress:
    """Tracks progress of a sync operation."""

    total_chunks: int
    completed: int
    failed: list[dict]
    last_success: float | None


# Custom exception classes for error handling
class FulcraDBError(Exception):
    """Base exception for database errors."""

    pass


class DatabaseNotEnabledError(FulcraDBError):
    """Database is not enabled."""

    pass


class HealthDatabase:
    """
    SQLite-based personal health data warehouse.

    This is NOT a cache - it's permanent storage for your health data.
    Data is fetched once from Fulcra and stored forever locally.
    """

    # Default recent data window (data newer than this might still be updating)
    RECENT_DATA_HOURS = 2

    # Expected sample intervals for different metric types (in seconds)
    METRIC_INTERVALS = {
        "heart_rate": 60,  # 1 minute
        "steps": 60,
        "active_energy": 60,
        "default": 60,
    }

    def __init__(self, db_path: str | Path | None = None, enabled: bool = True):
        """
        Initialize the health database.

        Args:
            db_path: Path to the SQLite database file. If None, uses default location.
            enabled: If False, all operations become no-ops.
        """
        self.enabled = enabled
        if not enabled:
            self.db_path = None
            return

        if db_path is None:
            # Default to ~/.fulcra_health_db/fulcra_health.db
            default_dir = Path.home() / ".fulcra_health_db"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = default_dir / "fulcra_health.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize database schema with performance optimizations."""
        with self._get_connection() as conn:
            # Enable performance optimizations
            conn.execute("PRAGMA page_size = 4096")
            conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O

            conn.executescript(
                """
                -- Core health metrics table (time-series data)
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL,
                    value_json TEXT,
                    source TEXT DEFAULT 'fulcra_api',
                    fetched_at REAL NOT NULL,
                    metadata TEXT,
                    UNIQUE(metric_name, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_metric_time
                    ON health_metrics(metric_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON health_metrics(timestamp);

                -- Workouts table
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workout_id TEXT UNIQUE,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    workout_type TEXT,
                    data_json TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workout_time
                    ON workouts(start_time, end_time);

                -- Sleep cycles table
                CREATE TABLE IF NOT EXISTS sleep_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT UNIQUE,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    data_json TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sleep_time
                    ON sleep_cycles(start_time, end_time);

                -- Location data table
                CREATE TABLE IF NOT EXISTS locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    accuracy REAL,
                    data_json TEXT,
                    fetched_at REAL NOT NULL,
                    UNIQUE(timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_location_time
                    ON locations(timestamp);

                -- Metadata cache (for rarely-changing data like user info, metrics catalog)
                CREATE TABLE IF NOT EXISTS metadata_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    expires_at REAL
                );

                -- Query/fetch log (track what we've fetched and when)
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    query_params TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    fetched_at REAL NOT NULL,
                    record_count INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_fetch_log_tool
                    ON fetch_log(tool_name, start_time, end_time);

                -- Sync jobs table (for resume capability)
                CREATE TABLE IF NOT EXISTS sync_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    start_date REAL NOT NULL,
                    end_date REAL NOT NULL,
                    status TEXT NOT NULL,
                    progress_percent REAL,
                    last_synced_date REAL,
                    chunks_completed INTEGER,
                    chunks_total INTEGER,
                    chunks_failed TEXT,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    error_message TEXT
                );

                -- Error log table
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    context TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_at REAL
                );

                -- Enable WAL mode for better concurrency
                PRAGMA journal_mode=WAL;
            """
            )

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        if not self.enabled:
            raise DatabaseNotEnabledError("Health database is not enabled")

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Timestamp conversion utilities
    # =========================================================================

    @staticmethod
    def to_timestamp(dt: datetime | float | str) -> float:
        """Convert various datetime formats to Unix timestamp."""
        if isinstance(dt, (int, float)):
            return float(dt)
        if isinstance(dt, str):
            # Parse ISO8601 string
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if isinstance(dt, datetime):
            return dt.timestamp()
        raise ValueError(f"Cannot convert {type(dt)} to timestamp")

    @staticmethod
    def from_timestamp(ts: float) -> datetime:
        """Convert Unix timestamp to datetime (UTC)."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    # =========================================================================
    # Health Metrics (time-series data)
    # =========================================================================

    def insert_metrics(
        self,
        metric_name: str,
        data: list[dict],
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> int:
        """
        Insert metric time-series data.

        Args:
            metric_name: Name of the metric (e.g., "heart_rate")
            data: List of data points with timestamp and value
            timestamp_field: Name of the timestamp field in data
            value_field: Name of the value field in data

        Returns:
            Number of records inserted
        """
        if not self.enabled or not data:
            return 0

        now = time.time()
        records = []

        for point in data:
            ts = self.to_timestamp(point.get(timestamp_field))
            value = point.get(value_field)

            # Handle complex values (store as JSON)
            value_json = None
            if isinstance(value, (dict, list)):
                value_json = json.dumps(value, default=str)
                value = None

            # Store any extra fields as metadata
            extra_fields = {
                k: v for k, v in point.items() if k not in (timestamp_field, value_field)
            }
            metadata = json.dumps(extra_fields, default=str) if extra_fields else None

            records.append((metric_name, ts, value, value_json, now, metadata))

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO health_metrics
                    (metric_name, timestamp, value, value_json, fetched_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            inserted = cursor.rowcount

        logger.info(
            "Inserted metrics",
            metric_name=metric_name,
            total=len(data),
            inserted=inserted,
        )
        return inserted

    def upsert_metrics(
        self,
        metric_name: str,
        data: list[dict],
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> int:
        """
        Insert or update metric data (for recent data that might change).

        Args:
            metric_name: Name of the metric
            data: List of data points
            timestamp_field: Name of timestamp field
            value_field: Name of value field

        Returns:
            Number of records affected
        """
        if not self.enabled or not data:
            return 0

        now = time.time()
        records = []

        for point in data:
            ts = self.to_timestamp(point.get(timestamp_field))
            value = point.get(value_field)

            value_json = None
            if isinstance(value, (dict, list)):
                value_json = json.dumps(value, default=str)
                value = None

            extra_fields = {
                k: v for k, v in point.items() if k not in (timestamp_field, value_field)
            }
            metadata = json.dumps(extra_fields, default=str) if extra_fields else None

            records.append((metric_name, ts, value, value_json, now, metadata))

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR REPLACE INTO health_metrics
                    (metric_name, timestamp, value, value_json, fetched_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            affected = cursor.rowcount

        return affected

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> list[dict]:
        """
        Query metrics for a time range.

        Args:
            metric_name: Name of the metric
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)

        Returns:
            List of data points
        """
        if not self.enabled:
            return []

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, value, value_json, metadata
                FROM health_metrics
                WHERE metric_name = ?
                  AND timestamp >= ?
                  AND timestamp < ?
                ORDER BY timestamp ASC
            """,
                (metric_name, start_ts, end_ts),
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            point = {
                "timestamp": self.from_timestamp(row["timestamp"]).isoformat(),
            }

            if row["value_json"]:
                point["value"] = json.loads(row["value_json"])
            else:
                point["value"] = row["value"]

            if row["metadata"]:
                point.update(json.loads(row["metadata"]))

            results.append(point)

        return results

    def get_metric_timestamps(
        self,
        metric_name: str,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> list[float]:
        """Get just the timestamps for a metric (for gap detection)."""
        if not self.enabled:
            return []

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp FROM health_metrics
                WHERE metric_name = ?
                  AND timestamp >= ?
                  AND timestamp < ?
                ORDER BY timestamp ASC
            """,
                (metric_name, start_ts, end_ts),
            )
            return [row["timestamp"] for row in cursor.fetchall()]

    # =========================================================================
    # Gap Detection
    # =========================================================================

    def identify_gaps(
        self,
        metric_name: str,
        start_time: datetime | float,
        end_time: datetime | float,
        expected_interval: float | None = None,
    ) -> list[TimeGap]:
        """
        Identify time gaps in local data that need to be fetched.

        Args:
            metric_name: Name of the metric
            start_time: Start of requested range
            end_time: End of requested range
            expected_interval: Expected seconds between samples (auto-detected if None)

        Returns:
            List of TimeGap objects representing missing data ranges
        """
        if not self.enabled:
            # If disabled, treat entire range as a gap
            return [
                TimeGap(
                    self.to_timestamp(start_time),
                    self.to_timestamp(end_time),
                )
            ]

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        # Get local timestamps
        timestamps = self.get_metric_timestamps(metric_name, start_ts, end_ts)

        if not timestamps:
            # No local data - entire range is a gap
            return [TimeGap(start_ts, end_ts)]

        # Determine expected interval
        if expected_interval is None:
            expected_interval = self.METRIC_INTERVALS.get(
                metric_name, self.METRIC_INTERVALS["default"]
            )

        # Allow some tolerance (2x expected interval considered a gap)
        gap_threshold = expected_interval * 2

        gaps = []

        # Check for gap before first local record
        if timestamps[0] > start_ts + gap_threshold:
            gaps.append(TimeGap(start_ts, timestamps[0]))

        # Check for gaps between local records
        for i in range(len(timestamps) - 1):
            gap_size = timestamps[i + 1] - timestamps[i]
            if gap_size > gap_threshold:
                gaps.append(TimeGap(timestamps[i], timestamps[i + 1]))

        # Check for gap after last local record
        if timestamps[-1] < end_ts - gap_threshold:
            gaps.append(TimeGap(timestamps[-1], end_ts))

        return gaps

    def is_recent_data(self, timestamp: datetime | float) -> bool:
        """Check if a timestamp is within the 'recent data' window."""
        ts = self.to_timestamp(timestamp)
        cutoff = time.time() - (self.RECENT_DATA_HOURS * 3600)
        return ts > cutoff

    # =========================================================================
    # Workouts
    # =========================================================================

    def insert_workouts(self, workouts: list[dict]) -> int:
        """Insert workout records."""
        if not self.enabled or not workouts:
            return 0

        now = time.time()
        records = []

        for workout in workouts:
            # Get time values - skip if None
            start_val = workout.get("start_time") or workout.get("startDate")
            end_val = workout.get("end_time") or workout.get("endDate")

            if start_val is None or end_val is None:
                logger.warning(
                    "Skipping workout with missing time data",
                    has_start=start_val is not None,
                    has_end=end_val is not None,
                )
                continue

            # Generate a unique ID if not provided
            workout_id = workout.get("id") or workout.get("workout_id")
            if not workout_id:
                # Create hash from start time and type
                hash_input = f"{start_val}-{workout.get('workout_type', 'unknown')}"
                workout_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

            start_time = self.to_timestamp(start_val)
            end_time = self.to_timestamp(end_val)
            workout_type = workout.get("workout_type") or workout.get("workoutActivityType")

            records.append(
                (
                    workout_id,
                    start_time,
                    end_time,
                    workout_type,
                    json.dumps(workout, default=str),  # Use default=str to handle datetime objects
                    now,
                )
            )

        if not records:
            return 0

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO workouts
                    (workout_id, start_time, end_time, workout_type, data_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            return cursor.rowcount

    def query_workouts(
        self,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> list[dict]:
        """Query workouts for a time range."""
        if not self.enabled:
            return []

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT data_json FROM workouts
                WHERE start_time < ? AND end_time > ?
                ORDER BY start_time ASC
            """,
                (end_ts, start_ts),
            )
            return [json.loads(row["data_json"]) for row in cursor.fetchall()]

    def has_workouts_for_range(
        self,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> bool:
        """Check if we have fetched workouts for a time range."""
        if not self.enabled:
            return False

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        # Check fetch log for this range
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM fetch_log
                WHERE tool_name = 'get_workouts'
                  AND start_time <= ?
                  AND end_time >= ?
                LIMIT 1
            """,
                (start_ts, end_ts),
            )
            return cursor.fetchone() is not None

    # =========================================================================
    # Sleep Cycles
    # =========================================================================

    def insert_sleep_cycles(self, cycles: list[dict]) -> int:
        """Insert sleep cycle records."""
        if not self.enabled or not cycles:
            return 0

        now = time.time()
        records = []

        for cycle in cycles:
            # Get time values - skip if None
            start_val = cycle.get("start_time") or cycle.get("startDate")
            end_val = cycle.get("end_time") or cycle.get("endDate")

            if start_val is None or end_val is None:
                logger.warning(
                    "Skipping sleep cycle with missing time data",
                    has_start=start_val is not None,
                    has_end=end_val is not None,
                )
                continue

            cycle_id = cycle.get("id") or cycle.get("cycle_id")
            if not cycle_id:
                hash_input = f"{start_val}-{end_val}"
                cycle_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

            start_time = self.to_timestamp(start_val)
            end_time = self.to_timestamp(end_val)

            records.append(
                (
                    cycle_id,
                    start_time,
                    end_time,
                    json.dumps(cycle, default=str),  # Use default=str to handle datetime objects
                    now,
                )
            )

        if not records:
            return 0

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO sleep_cycles
                    (cycle_id, start_time, end_time, data_json, fetched_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                records,
            )
            return cursor.rowcount

    def query_sleep_cycles(
        self,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> list[dict]:
        """Query sleep cycles for a time range."""
        if not self.enabled:
            return []

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT data_json FROM sleep_cycles
                WHERE start_time < ? AND end_time > ?
                ORDER BY start_time ASC
            """,
                (end_ts, start_ts),
            )
            return [json.loads(row["data_json"]) for row in cursor.fetchall()]

    # =========================================================================
    # Locations
    # =========================================================================

    def insert_locations(self, locations: list[dict]) -> int:
        """Insert location records."""
        if not self.enabled or not locations:
            return 0

        now = time.time()
        records = []

        for loc in locations:
            ts = self.to_timestamp(loc.get("timestamp") or loc.get("time"))
            lat = loc.get("latitude") or loc.get("lat")
            lon = loc.get("longitude") or loc.get("lon") or loc.get("lng")
            accuracy = loc.get("accuracy") or loc.get("horizontal_accuracy")

            records.append(
                (
                    ts,
                    lat,
                    lon,
                    accuracy,
                    json.dumps(loc, default=str),
                    now,
                )
            )

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO locations
                    (timestamp, latitude, longitude, accuracy, data_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            return cursor.rowcount

    def query_locations(
        self,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> list[dict]:
        """Query locations for a time range."""
        if not self.enabled:
            return []

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT data_json FROM locations
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
            """,
                (start_ts, end_ts),
            )
            return [json.loads(row["data_json"]) for row in cursor.fetchall()]

    def get_location_at_time(
        self,
        target_time: datetime | float,
        window_size: int = 14400,
    ) -> dict | None:
        """Get closest location to a specific time."""
        if not self.enabled:
            return None

        target_ts = self.to_timestamp(target_time)

        with self._get_connection() as conn:
            # Look for closest location within window
            cursor = conn.execute(
                """
                SELECT data_json, ABS(timestamp - ?) as time_diff
                FROM locations
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY time_diff ASC
                LIMIT 1
            """,
                (target_ts, target_ts - window_size, target_ts + window_size),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row["data_json"])
            return None

    # =========================================================================
    # Metadata Cache
    # =========================================================================

    def get_metadata(self, key: str, max_age: float | None = None) -> Any | None:
        """
        Get cached metadata.

        Args:
            key: Cache key
            max_age: Maximum age in seconds (None = no limit)

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT value, fetched_at, expires_at FROM metadata_cache
                WHERE key = ?
            """,
                (key,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Check explicit expiration
            if row["expires_at"] and row["expires_at"] < time.time():
                return None

            # Check max_age
            if max_age and (time.time() - row["fetched_at"]) > max_age:
                return None

            return json.loads(row["value"])

    def set_metadata(
        self,
        key: str,
        value: Any,
        expires_in: float | None = None,
    ):
        """
        Set cached metadata.

        Args:
            key: Cache key
            value: Value to cache
            expires_in: Seconds until expiration (None = never)
        """
        if not self.enabled:
            return

        now = time.time()
        expires_at = now + expires_in if expires_in else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata_cache
                    (key, value, fetched_at, expires_at)
                VALUES (?, ?, ?, ?)
            """,
                (key, json.dumps(value, default=str), now, expires_at),
            )

    # =========================================================================
    # Fetch Logging
    # =========================================================================

    def log_fetch(
        self,
        tool_name: str,
        query_params: dict,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        record_count: int | None = None,
    ):
        """Log a fetch operation."""
        if not self.enabled:
            return

        start_ts = self.to_timestamp(start_time) if start_time else None
        end_ts = self.to_timestamp(end_time) if end_time else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fetch_log
                    (tool_name, query_params, start_time, end_time, fetched_at, record_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (tool_name, json.dumps(query_params, default=str), start_ts, end_ts, time.time(), record_count),
            )

    def was_range_fetched(
        self,
        tool_name: str,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> bool:
        """Check if a time range was previously fetched."""
        if not self.enabled:
            return False

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM fetch_log
                WHERE tool_name = ?
                  AND start_time <= ?
                  AND end_time >= ?
                LIMIT 1
            """,
                (tool_name, start_ts, end_ts),
            )
            return cursor.fetchone() is not None

    # =========================================================================
    # Error Logging
    # =========================================================================

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: dict | None = None,
    ):
        """Log an error to the database."""
        if not self.enabled:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO error_log
                    (timestamp, error_type, error_message, context)
                VALUES (?, ?, ?, ?)
            """,
                (
                    time.time(),
                    error_type,
                    error_message,
                    json.dumps(context, default=str) if context else None,
                ),
            )

    # =========================================================================
    # Database Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """Get comprehensive database statistics."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_connection() as conn:
            stats = {"enabled": True, "db_path": str(self.db_path)}

            # Database size
            if self.db_path.exists():
                stats["database_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

            # Record counts
            for table in [
                "health_metrics",
                "workouts",
                "sleep_cycles",
                "locations",
                "metadata_cache",
            ]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Metrics breakdown
            cursor = conn.execute(
                """
                SELECT metric_name, COUNT(*) as count,
                       MIN(timestamp) as oldest,
                       MAX(timestamp) as newest
                FROM health_metrics
                GROUP BY metric_name
            """
            )
            stats["metrics_by_name"] = {
                row["metric_name"]: {
                    "count": row["count"],
                    "oldest": self.from_timestamp(row["oldest"]).isoformat()
                    if row["oldest"]
                    else None,
                    "newest": self.from_timestamp(row["newest"]).isoformat()
                    if row["newest"]
                    else None,
                }
                for row in cursor.fetchall()
            }

            # Overall date range
            cursor = conn.execute(
                """
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                FROM health_metrics
            """
            )
            row = cursor.fetchone()
            if row["oldest"]:
                stats["date_range"] = {
                    "oldest": self.from_timestamp(row["oldest"]).isoformat(),
                    "newest": self.from_timestamp(row["newest"]).isoformat(),
                }

            # Last fetch time
            cursor = conn.execute(
                """
                SELECT MAX(fetched_at) as last_fetch FROM fetch_log
            """
            )
            row = cursor.fetchone()
            if row["last_fetch"]:
                stats["last_sync"] = self.from_timestamp(row["last_fetch"]).isoformat()

            return stats

    # =========================================================================
    # Data Export
    # =========================================================================

    def export_metrics_csv(
        self,
        output_path: str | Path,
        metric_name: str | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
    ) -> int:
        """
        Export health metrics to CSV.

        Args:
            output_path: Path for output CSV file
            metric_name: Optional filter by metric name
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Number of records exported
        """
        if not self.enabled:
            return 0

        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        query = "SELECT metric_name, timestamp, value, value_json, metadata FROM health_metrics"
        params = []
        conditions = []

        if metric_name:
            conditions.append("metric_name = ?")
            params.append(metric_name)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(self.to_timestamp(start_time))
        if end_time:
            conditions.append("timestamp < ?")
            params.append(self.to_timestamp(end_time))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY metric_name, timestamp"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_name", "timestamp", "value", "value_json", "metadata"])

            for row in rows:
                writer.writerow(
                    [
                        row["metric_name"],
                        self.from_timestamp(row["timestamp"]).isoformat(),
                        row["value"],
                        row["value_json"],
                        row["metadata"],
                    ]
                )

        return len(rows)

    def export_json(
        self,
        output_path: str | Path,
        include_metrics: bool = True,
        include_workouts: bool = True,
        include_sleep: bool = True,
        include_locations: bool = True,
    ) -> dict:
        """
        Export all data to JSON.

        Args:
            output_path: Path for output JSON file
            include_*: Flags to control what data to export

        Returns:
            Summary of exported data
        """
        if not self.enabled:
            return {"enabled": False}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {"exported_at": datetime.now(timezone.utc).isoformat()}
        summary = {}

        with self._get_connection() as conn:
            if include_metrics:
                cursor = conn.execute(
                    """
                    SELECT metric_name, timestamp, value, value_json, metadata
                    FROM health_metrics ORDER BY timestamp
                """
                )
                export_data["health_metrics"] = [
                    {
                        "metric_name": r["metric_name"],
                        "timestamp": self.from_timestamp(r["timestamp"]).isoformat(),
                        "value": json.loads(r["value_json"]) if r["value_json"] else r["value"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                    }
                    for r in cursor.fetchall()
                ]
                summary["health_metrics"] = len(export_data["health_metrics"])

            if include_workouts:
                cursor = conn.execute("SELECT data_json FROM workouts ORDER BY start_time")
                export_data["workouts"] = [json.loads(r["data_json"]) for r in cursor.fetchall()]
                summary["workouts"] = len(export_data["workouts"])

            if include_sleep:
                cursor = conn.execute("SELECT data_json FROM sleep_cycles ORDER BY start_time")
                export_data["sleep_cycles"] = [
                    json.loads(r["data_json"]) for r in cursor.fetchall()
                ]
                summary["sleep_cycles"] = len(export_data["sleep_cycles"])

            if include_locations:
                cursor = conn.execute("SELECT data_json FROM locations ORDER BY timestamp")
                export_data["locations"] = [json.loads(r["data_json"]) for r in cursor.fetchall()]
                summary["locations"] = len(export_data["locations"])

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return summary

    # =========================================================================
    # Database Management
    # =========================================================================

    def clear(
        self,
        older_than_days: int | None = None,
        tables: list[str] | None = None,
    ) -> dict:
        """
        Clear data from the database.

        Args:
            older_than_days: Only clear data older than this many days
            tables: List of tables to clear (None = all)

        Returns:
            Count of deleted records per table
        """
        if not self.enabled:
            return {"enabled": False}

        all_tables = ["health_metrics", "workouts", "sleep_cycles", "locations"]
        tables_to_clear = tables if tables else all_tables

        deleted = {}
        cutoff_ts = None
        if older_than_days:
            cutoff_ts = time.time() - (older_than_days * 86400)

        with self._get_connection() as conn:
            for table in tables_to_clear:
                if table not in all_tables:
                    continue

                if cutoff_ts:
                    if table == "health_metrics":
                        cursor = conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_ts,)
                        )
                    else:
                        cursor = conn.execute(
                            f"DELETE FROM {table} WHERE start_time < ?", (cutoff_ts,)
                        )
                else:
                    cursor = conn.execute(f"DELETE FROM {table}")

                deleted[table] = cursor.rowcount

        # Vacuum to reclaim space (must be outside transaction)
        with self._get_connection() as conn:
            conn.execute("VACUUM")

        return deleted

    def close(self):
        """Close database (no-op, connections are managed per-operation)."""
        pass

    # =========================================================================
    # Batch Operations (Performance Optimizations)
    # =========================================================================

    def batch_insert_metrics(
        self,
        data_by_metric: dict[str, list[dict]],
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> dict[str, int]:
        """
        Insert metrics for multiple metric types in a single transaction.

        This is more efficient than calling insert_metrics multiple times
        when you have data for multiple metrics.

        Args:
            data_by_metric: Dict mapping metric_name to list of data points
            timestamp_field: Name of timestamp field in data
            value_field: Name of value field in data

        Returns:
            Dict mapping metric_name to count of inserted records
        """
        if not self.enabled or not data_by_metric:
            return {}

        now = time.time()
        all_records = []
        counts = {}

        for metric_name, data in data_by_metric.items():
            metric_records = []
            for point in data:
                ts = self.to_timestamp(point.get(timestamp_field))
                value = point.get(value_field)

                value_json = None
                if isinstance(value, (dict, list)):
                    value_json = json.dumps(value, default=str)
                    value = None

                extra_fields = {
                    k: v for k, v in point.items() if k not in (timestamp_field, value_field)
                }
                metadata = json.dumps(extra_fields, default=str) if extra_fields else None

                metric_records.append((metric_name, ts, value, value_json, now, metadata))

            all_records.extend(metric_records)
            counts[metric_name] = len(metric_records)

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO health_metrics
                    (metric_name, timestamp, value, value_json, fetched_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                all_records,
            )

        return counts

    def query_multiple_metrics(
        self,
        metric_names: list[str],
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> dict[str, list[dict]]:
        """
        Query multiple metrics in a single database operation.

        More efficient than querying metrics one by one.

        Args:
            metric_names: List of metric names to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict mapping metric_name to list of data points
        """
        if not self.enabled or not metric_names:
            return {}

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        # Build query with placeholders for metric names
        placeholders = ",".join("?" * len(metric_names))

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT metric_name, timestamp, value, value_json, metadata
                FROM health_metrics
                WHERE metric_name IN ({placeholders})
                  AND timestamp >= ?
                  AND timestamp < ?
                ORDER BY metric_name, timestamp ASC
            """,
                (*metric_names, start_ts, end_ts),
            )
            rows = cursor.fetchall()

        # Group results by metric name
        results: dict[str, list[dict]] = {name: [] for name in metric_names}

        for row in rows:
            point = {
                "timestamp": self.from_timestamp(row["timestamp"]).isoformat(),
            }

            if row["value_json"]:
                point["value"] = json.loads(row["value_json"])
            else:
                point["value"] = row["value"]

            if row["metadata"]:
                point.update(json.loads(row["metadata"]))

            results[row["metric_name"]].append(point)

        return results

    def get_coverage_summary(
        self,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> dict:
        """
        Get a summary of data coverage for a time range.

        Useful for understanding what data needs to be fetched.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict with coverage information per data type
        """
        if not self.enabled:
            return {"enabled": False}

        start_ts = self.to_timestamp(start_time)
        end_ts = self.to_timestamp(end_time)

        with self._get_connection() as conn:
            coverage = {
                "time_range": {
                    "start": self.from_timestamp(start_ts).isoformat(),
                    "end": self.from_timestamp(end_ts).isoformat(),
                },
                "metrics": {},
                "workouts": 0,
                "sleep_cycles": 0,
                "locations": 0,
            }

            # Metrics coverage
            cursor = conn.execute(
                """
                SELECT metric_name, COUNT(*) as count,
                       MIN(timestamp) as first, MAX(timestamp) as last
                FROM health_metrics
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY metric_name
            """,
                (start_ts, end_ts),
            )
            for row in cursor.fetchall():
                coverage["metrics"][row["metric_name"]] = {
                    "count": row["count"],
                    "first": self.from_timestamp(row["first"]).isoformat(),
                    "last": self.from_timestamp(row["last"]).isoformat(),
                }

            # Workouts count
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM workouts
                WHERE start_time < ? AND end_time > ?
            """,
                (end_ts, start_ts),
            )
            coverage["workouts"] = cursor.fetchone()[0]

            # Sleep cycles count
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM sleep_cycles
                WHERE start_time < ? AND end_time > ?
            """,
                (end_ts, start_ts),
            )
            coverage["sleep_cycles"] = cursor.fetchone()[0]

            # Locations count
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM locations
                WHERE timestamp >= ? AND timestamp < ?
            """,
                (start_ts, end_ts),
            )
            coverage["locations"] = cursor.fetchone()[0]

            return coverage


# Module-level singleton for easy access
_db_instance: HealthDatabase | None = None


def get_health_db() -> HealthDatabase:
    """Get the global health database instance."""
    global _db_instance

    if _db_instance is None:
        # Check environment variables for configuration
        enabled = os.environ.get("FULCRA_DB_ENABLED", "true").lower() == "true"
        db_path = os.environ.get("FULCRA_DB_PATH")

        _db_instance = HealthDatabase(
            db_path=db_path,
            enabled=enabled,
        )

    return _db_instance


def reset_health_db():
    """Reset the global health database instance (for testing)."""
    global _db_instance
    _db_instance = None
