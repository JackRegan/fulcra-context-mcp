"""
Smart Fetch Module - Intelligent Data Fetching with Local-First Strategy

This module provides wrappers around Fulcra API calls that:
1. Check local database first
2. Only fetch missing data (gaps)
3. Handle recent data specially (last 2 hours might still be updating)
4. Support chunked requests for large date ranges
5. Implement retry logic with exponential backoff
6. Provide resume capability for interrupted syncs
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable

import structlog

from .health_db import HealthDatabase, TimeGap, get_health_db

logger = structlog.getLogger(__name__)


# Configuration from environment
CHUNK_SIZE_DAYS = int(os.environ.get("FULCRA_CHUNK_SIZE_DAYS", "7"))
MAX_RETRIES = int(os.environ.get("FULCRA_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("FULCRA_RETRY_BASE_DELAY", "1.0"))
REQUEST_TIMEOUT = int(os.environ.get("FULCRA_REQUEST_TIMEOUT", "30"))
RATE_LIMIT_DELAY = float(os.environ.get("FULCRA_RATE_LIMIT_DELAY", "0.5"))
RECENT_DATA_HOURS = int(os.environ.get("FULCRA_DB_RECENT_DATA_HOURS", "2"))


# Custom exception classes
class FulcraAPIError(Exception):
    """Base exception for Fulcra API errors."""

    pass


class RateLimitError(FulcraAPIError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after or 60


class AuthenticationError(FulcraAPIError):
    """Authentication failed or token expired."""

    pass


class DataNotFoundError(FulcraAPIError):
    """Requested data doesn't exist."""

    pass


class APITimeoutError(FulcraAPIError):
    """Request timed out."""

    pass


class InvalidParameterError(FulcraAPIError):
    """Invalid parameters in request."""

    pass


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    data: Any
    from_cache: bool
    gaps_fetched: int = 0
    records_fetched: int = 0
    records_from_cache: int = 0


@dataclass
class SyncJobProgress:
    """Tracks progress of a large sync operation."""

    job_id: str
    metric_name: str
    start_date: datetime
    end_date: datetime
    total_chunks: int
    completed: int = 0
    failed: list[dict] = field(default_factory=list)
    last_success: datetime | None = None
    status: str = "running"


def to_timestamp(dt: datetime | float | str) -> float:
    """Convert datetime to Unix timestamp."""
    if isinstance(dt, (int, float)):
        return float(dt)
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if isinstance(dt, datetime):
        return dt.timestamp()
    raise ValueError(f"Cannot convert {type(dt)} to timestamp")


def from_timestamp(ts: float) -> datetime:
    """Convert Unix timestamp to datetime (UTC)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _convert_timestamps_in_records(records: list[dict]) -> list[dict]:
    """Convert pandas/numpy types to JSON-serializable Python types."""
    import numpy as np
    import pandas as pd

    def convert_value(value):
        """Recursively convert a value to JSON-serializable type."""
        if value is None:
            return None
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif hasattr(value, "isoformat"):  # datetime-like objects
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()  # Convert numpy scalar to Python scalar
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [convert_value(v) for v in value]
        else:
            return value

    converted = []
    for record in records:
        new_record = {key: convert_value(value) for key, value in record.items()}
        converted.append(new_record)
    return converted


def _normalize_metric_records(records: list[dict]) -> list[dict]:
    """
    Normalize Fulcra API field names to our expected format.

    Fulcra returns: {"start_date": "...", "end_date": "...", "value": ...}
    We expect: {"timestamp": <float>, "value": ...}

    Records without valid timestamps are filtered out.
    """
    normalized = []
    for record in records:
        new_record = dict(record)  # Copy original

        # Map start_date -> timestamp if timestamp doesn't exist
        if "timestamp" not in new_record:
            timestamp_value = None
            if "start_date" in new_record:
                timestamp_value = new_record["start_date"]
            elif "startDate" in new_record:
                timestamp_value = new_record["startDate"]
            elif "time" in new_record:
                timestamp_value = new_record["time"]

            # Convert to float timestamp
            if timestamp_value is not None:
                new_record["timestamp"] = to_timestamp(timestamp_value)
            else:
                # Skip records without valid timestamps
                logger.warning(
                    "Skipping record without timestamp",
                    record_keys=list(record.keys()),
                    sample_values={k: str(v)[:50] for k, v in list(record.items())[:3]},
                )
                continue

        # Also verify timestamp field exists and is not None
        if "timestamp" not in new_record or new_record["timestamp"] is None:
            logger.warning(
                "Skipping record with None timestamp",
                record_keys=list(record.keys()),
            )
            continue

        normalized.append(new_record)
    return normalized


def _normalize_workout_records(records: list[dict]) -> list[dict]:
    """
    Normalize workout records, filtering out invalid entries.

    Converts start_date/end_date to start_time/end_time as float timestamps.
    """
    valid = []
    for record in records:
        new_record = dict(record)  # Copy original

        # Get start time from various possible field names
        start = (
            record.get("start_time")
            or record.get("start_date")
            or record.get("startTime")
            or record.get("startDate")
        )

        # Get end time from various possible field names
        end = (
            record.get("end_time")
            or record.get("end_date")
            or record.get("endTime")
            or record.get("endDate")
        )

        if start is None or end is None:
            logger.warning("Skipping workout with missing time data", record_keys=list(record.keys()))
            continue

        # Convert to float timestamps and set normalized field names
        new_record["start_time"] = to_timestamp(start)
        new_record["end_time"] = to_timestamp(end)

        valid.append(new_record)
    return valid


def _normalize_sleep_records(records: list[dict]) -> list[dict]:
    """
    Normalize sleep cycle records, filtering out invalid entries.

    Converts start_date/end_date to start_time/end_time as float timestamps.
    """
    valid = []
    for record in records:
        new_record = dict(record)  # Copy original

        # Get start time from various possible field names
        start = (
            record.get("start_time")
            or record.get("start_date")
            or record.get("startTime")
            or record.get("startDate")
        )

        # Get end time from various possible field names
        end = (
            record.get("end_time")
            or record.get("end_date")
            or record.get("endTime")
            or record.get("endDate")
        )

        if start is None or end is None:
            logger.warning("Skipping sleep cycle with missing time data", record_keys=list(record.keys()))
            continue

        # Convert to float timestamps and set normalized field names
        new_record["start_time"] = to_timestamp(start)
        new_record["end_time"] = to_timestamp(end)

        valid.append(new_record)
    return valid


def split_into_chunks(
    start_date: datetime,
    end_date: datetime,
    chunk_size_days: int = CHUNK_SIZE_DAYS,
) -> list[tuple[datetime, datetime]]:
    """
    Split a date range into manageable chunks.

    Args:
        start_date: Start of range
        end_date: End of range
        chunk_size_days: Size of each chunk in days

    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_size_days), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end

    return chunks


async def fetch_with_retry(
    fetch_func: Callable,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
    **kwargs,
) -> Any:
    """
    Execute a fetch function with exponential backoff retry logic.

    Args:
        fetch_func: The function to call
        *args: Positional arguments for fetch_func
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        **kwargs: Keyword arguments for fetch_func

    Returns:
        Result from fetch_func

    Raises:
        Various FulcraAPIError subclasses on unrecoverable errors
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # If it's an async function, await it
            if asyncio.iscoroutinefunction(fetch_func):
                return await fetch_func(*args, **kwargs)
            else:
                return fetch_func(*args, **kwargs)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            delay = max(base_delay * (2**attempt), e.retry_after or 1)
            logger.warning(
                "Rate limited, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
            )
            await asyncio.sleep(delay)
            last_error = e

        except AuthenticationError:
            # Auth errors should be handled by re-authenticating
            # Don't retry here, let the caller handle it
            raise

        except (TimeoutError, ConnectionError, OSError) as e:
            if attempt == max_retries - 1:
                raise APITimeoutError(f"Request failed after {max_retries} attempts: {e}")

            delay = base_delay * (2**attempt)
            logger.warning(
                "Network error, retrying",
                error=str(e),
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
            )
            await asyncio.sleep(delay)
            last_error = e

        except Exception as e:
            # Unknown error - don't retry
            logger.error("Unrecoverable error", error=str(e))
            raise

    # Should not reach here, but just in case
    raise last_error or FulcraAPIError("Fetch failed after all retries")


class SmartFetcher:
    """
    Smart data fetcher that uses local database as primary storage.

    Implements the "fetch once, store forever" strategy with intelligent
    gap detection to minimize API calls.
    """

    def __init__(self, db: HealthDatabase | None = None):
        """
        Initialize the smart fetcher.

        Args:
            db: Health database instance (uses global singleton if None)
        """
        self.db = db or get_health_db()

    def _is_recent(self, timestamp: datetime | float) -> bool:
        """Check if timestamp is within the recent data window."""
        ts = to_timestamp(timestamp)
        cutoff = time.time() - (RECENT_DATA_HOURS * 3600)
        return ts > cutoff

    async def get_metric_time_series(
        self,
        fulcra_fetch_func: Callable,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs,
    ) -> FetchResult:
        """
        Get metric time series with smart local-first fetching.

        Args:
            fulcra_fetch_func: The Fulcra API function to call for fetching
            metric_name: Name of the metric
            start_time: Start of time range
            end_time: End of time range
            **kwargs: Additional arguments for the API call

        Returns:
            FetchResult with data and fetch statistics
        """
        # If database is disabled, just fetch from API
        if not self.db.enabled:
            data = await fetch_with_retry(
                fulcra_fetch_func,
                metric=metric_name,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )
            return FetchResult(data=data, from_cache=False)

        # Identify gaps in local data
        gaps = self.db.identify_gaps(metric_name, start_time, end_time)

        logger.info(
            "DEBUG: Gaps identified",
            metric_name=metric_name,
            gaps_count=len(gaps),
            gaps=[(g.start_time, g.end_time) for g in gaps] if gaps else "no gaps",
        )

        records_fetched = 0
        gaps_fetched = 0
        all_fetched_records = []  # Accumulate all fetched records

        # Fetch data for each gap
        for gap in gaps:
            gap_start = from_timestamp(gap.start_time)
            gap_end = from_timestamp(gap.end_time)

            # Determine if this is recent data (might still be updating)
            is_recent = self._is_recent(gap.end_time)

            try:
                # Fetch from API
                api_data = await fetch_with_retry(
                    fulcra_fetch_func,
                    metric=metric_name,
                    start_time=gap_start,
                    end_time=gap_end,
                    **kwargs,
                )

                logger.info(
                    "DEBUG: API fetch completed",
                    metric_name=metric_name,
                    api_data_type=type(api_data).__name__,
                    is_dataframe=hasattr(api_data, "to_dict"),
                    is_list=isinstance(api_data, list),
                )

                # Convert DataFrame to list of dicts if needed
                if hasattr(api_data, "to_dict"):
                    # Use date_format="iso" to convert Timestamps to strings
                    records = api_data.to_dict(orient="records")
                    # Convert any remaining Timestamp objects to ISO strings
                    records = _convert_timestamps_in_records(records)
                    logger.info(
                        "DEBUG: DataFrame converted to records",
                        records_count=len(records),
                        sample_record=records[0] if records else None,
                    )
                else:
                    records = api_data if isinstance(api_data, list) else [api_data]
                    logger.info(
                        "DEBUG: Using API data directly",
                        records_count=len(records) if isinstance(records, list) else 1,
                    )

                # Map Fulcra field names to our expected format
                # Fulcra uses "start_date" but we store as "timestamp"
                records = _normalize_metric_records(records)

                logger.info(
                    "DEBUG: Records normalized",
                    records_count=len(records),
                    sample_keys=list(records[0].keys()) if records else [],
                    sample_record=records[0] if records else None,
                )

                # Store in database
                if is_recent:
                    # Recent data - use upsert in case it changes
                    inserted_count = self.db.upsert_metrics(metric_name, records)
                    logger.info(
                        "DEBUG: Upserted metrics to database",
                        attempted=len(records),
                        upserted=inserted_count,
                    )
                else:
                    # Historical data - insert only
                    inserted_count = self.db.insert_metrics(metric_name, records)
                    logger.info(
                        "DEBUG: Inserted metrics to database",
                        attempted=len(records),
                        inserted=inserted_count,
                    )

                # Log the fetch
                self.db.log_fetch(
                    tool_name="get_metric_time_series",
                    query_params={"metric_name": metric_name, **kwargs},
                    start_time=gap_start,
                    end_time=gap_end,
                    record_count=len(records),
                )

                # Track fetched records
                all_fetched_records.extend(records)
                records_fetched += len(records)
                gaps_fetched += 1

                # Small delay between chunks to respect rate limits
                if gaps_fetched < len(gaps):
                    await asyncio.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(
                    "Failed to fetch gap",
                    metric_name=metric_name,
                    gap_start=gap_start.isoformat(),
                    gap_end=gap_end.isoformat(),
                    error=str(e),
                )
                self.db.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={
                        "metric_name": metric_name,
                        "gap_start": gap.start_time,
                        "gap_end": gap.end_time,
                    },
                )
                # Continue with other gaps
                continue

        # If we fetched data, combine with any cached data and return
        if gaps_fetched > 0:
            # Query database to get complete dataset (cached + just inserted)
            local_data = self.db.query_metrics(metric_name, start_time, end_time)

            return FetchResult(
                data=local_data if local_data else all_fetched_records,  # Fallback to fetched if query fails
                from_cache=False,
                gaps_fetched=gaps_fetched,
                records_fetched=records_fetched,
                records_from_cache=len(local_data) - records_fetched if local_data else 0,
            )
        else:
            # No gaps, all data from cache
            local_data = self.db.query_metrics(metric_name, start_time, end_time)

            return FetchResult(
                data=local_data,
                from_cache=True,
                gaps_fetched=0,
                records_fetched=0,
                records_from_cache=len(local_data),
            )

    async def get_workouts(
        self,
        fulcra_fetch_func: Callable,
        start_time: datetime,
        end_time: datetime,
    ) -> FetchResult:
        """
        Get workouts with smart local-first fetching.

        Args:
            fulcra_fetch_func: The Fulcra API function to call
            start_time: Start of time range
            end_time: End of time range

        Returns:
            FetchResult with workout data
        """
        if not self.db.enabled:
            data = await fetch_with_retry(fulcra_fetch_func, start_time, end_time)
            return FetchResult(data=data, from_cache=False)

        # Check if we already fetched this range
        if self.db.was_range_fetched("get_workouts", start_time, end_time):
            # Get from local database
            local_data = self.db.query_workouts(start_time, end_time)
            return FetchResult(
                data=local_data,
                from_cache=True,
                records_from_cache=len(local_data),
            )

        # Fetch from API
        api_data = await fetch_with_retry(fulcra_fetch_func, start_time, end_time)

        # Store in database
        workouts = api_data if isinstance(api_data, list) else [api_data]

        # Filter out invalid workouts (missing time data)
        workouts = _normalize_workout_records(workouts)

        if workouts:
            self.db.insert_workouts(workouts)

        # Log the fetch
        self.db.log_fetch(
            tool_name="get_workouts",
            query_params={},
            start_time=start_time,
            end_time=end_time,
            record_count=len(workouts),
        )

        return FetchResult(
            data=workouts,
            from_cache=False,
            records_fetched=len(workouts),
        )

    async def get_sleep_cycles(
        self,
        fulcra_fetch_func: Callable,
        start_time: datetime,
        end_time: datetime,
        **kwargs,
    ) -> FetchResult:
        """
        Get sleep cycles with smart local-first fetching.

        Args:
            fulcra_fetch_func: The Fulcra API function to call
            start_time: Start of time range
            end_time: End of time range
            **kwargs: Additional arguments for the API call

        Returns:
            FetchResult with sleep cycle data
        """
        if not self.db.enabled:
            data = await fetch_with_retry(
                fulcra_fetch_func,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )
            return FetchResult(data=data, from_cache=False)

        # For sleep cycles, we check by date range
        # Since sleep cycles span nights, we need to be careful about boundaries
        if self.db.was_range_fetched("get_sleep_cycles", start_time, end_time):
            local_data = self.db.query_sleep_cycles(start_time, end_time)
            return FetchResult(
                data=local_data,
                from_cache=True,
                records_from_cache=len(local_data),
            )

        # Fetch from API
        api_data = await fetch_with_retry(
            fulcra_fetch_func,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

        # Convert DataFrame to list if needed
        if hasattr(api_data, "to_dict"):
            cycles = api_data.to_dict(orient="records")
            # Convert any pandas Timestamp objects to ISO strings
            cycles = _convert_timestamps_in_records(cycles)
        else:
            cycles = api_data if isinstance(api_data, list) else [api_data]

        # Filter out invalid sleep cycles (missing time data)
        cycles = _normalize_sleep_records(cycles)

        # Store in database
        self.db.insert_sleep_cycles(cycles)

        # Log the fetch
        self.db.log_fetch(
            tool_name="get_sleep_cycles",
            query_params=kwargs,
            start_time=start_time,
            end_time=end_time,
            record_count=len(cycles),
        )

        return FetchResult(
            data=cycles,
            from_cache=False,
            records_fetched=len(cycles),
        )

    async def get_location_time_series(
        self,
        fulcra_fetch_func: Callable,
        start_time: datetime,
        end_time: datetime,
        **kwargs,
    ) -> FetchResult:
        """
        Get location time series with local-first fetching.

        Args:
            fulcra_fetch_func: The Fulcra API function to call
            start_time: Start of time range
            end_time: End of time range
            **kwargs: Additional arguments for the API call

        Returns:
            FetchResult with location data
        """
        if not self.db.enabled:
            data = await fetch_with_retry(
                fulcra_fetch_func,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )
            return FetchResult(data=data, from_cache=False)

        # Locations are a bit different - they're point-in-time data
        # We'll use a similar approach to metrics with gap detection

        # For simplicity, check if range was fetched
        if self.db.was_range_fetched("get_location_time_series", start_time, end_time):
            local_data = self.db.query_locations(start_time, end_time)
            return FetchResult(
                data=local_data,
                from_cache=True,
                records_from_cache=len(local_data),
            )

        # Fetch from API
        api_data = await fetch_with_retry(
            fulcra_fetch_func,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

        locations = api_data if isinstance(api_data, list) else [api_data]

        # Store in database
        self.db.insert_locations(locations)

        # Log the fetch
        self.db.log_fetch(
            tool_name="get_location_time_series",
            query_params=kwargs,
            start_time=start_time,
            end_time=end_time,
            record_count=len(locations),
        )

        return FetchResult(
            data=locations,
            from_cache=False,
            records_fetched=len(locations),
        )

    async def get_location_at_time(
        self,
        fulcra_fetch_func: Callable,
        target_time: datetime,
        window_size: int = 14400,
        **kwargs,
    ) -> FetchResult:
        """
        Get location at a specific time with local-first fetching.

        Args:
            fulcra_fetch_func: The Fulcra API function to call
            target_time: The time to get location for
            window_size: How far back to look for location data
            **kwargs: Additional arguments for the API call

        Returns:
            FetchResult with location data
        """
        if not self.db.enabled:
            data = await fetch_with_retry(
                fulcra_fetch_func,
                time=target_time,
                window_size=window_size,
                **kwargs,
            )
            return FetchResult(data=data, from_cache=False)

        # Try to find in local database first
        local_data = self.db.get_location_at_time(target_time, window_size)

        if local_data:
            return FetchResult(
                data=local_data,
                from_cache=True,
                records_from_cache=1,
            )

        # Fetch from API
        api_data = await fetch_with_retry(
            fulcra_fetch_func,
            time=target_time,
            window_size=window_size,
            **kwargs,
        )

        # Store in database if we got data
        if api_data:
            self.db.insert_locations([api_data])

        return FetchResult(
            data=api_data,
            from_cache=False,
            records_fetched=1 if api_data else 0,
        )

    async def get_cached_metadata(
        self,
        key: str,
        fulcra_fetch_func: Callable,
        max_age: float = 86400,  # 24 hours default
        **kwargs,
    ) -> FetchResult:
        """
        Get metadata with caching.

        Args:
            key: Cache key for this metadata
            fulcra_fetch_func: Function to fetch if not cached
            max_age: Maximum age of cached data in seconds
            **kwargs: Arguments for fetch function

        Returns:
            FetchResult with metadata
        """
        if not self.db.enabled:
            data = await fetch_with_retry(fulcra_fetch_func, **kwargs)
            return FetchResult(data=data, from_cache=False)

        # Check cache
        cached = self.db.get_metadata(key, max_age=max_age)
        if cached is not None:
            return FetchResult(data=cached, from_cache=True, records_from_cache=1)

        # Fetch fresh
        data = await fetch_with_retry(fulcra_fetch_func, **kwargs)

        # Cache it
        self.db.set_metadata(key, data)

        return FetchResult(data=data, from_cache=False, records_fetched=1)

    async def sync_large_range(
        self,
        fulcra_fetch_func: Callable,
        metric_name: str,
        start_date: datetime,
        end_date: datetime,
        chunk_size_days: int = CHUNK_SIZE_DAYS,
        **kwargs,
    ) -> dict:
        """
        Sync a large date range with progress tracking.

        Args:
            fulcra_fetch_func: The Fulcra API function to call
            metric_name: Name of the metric to sync
            start_date: Start of date range
            end_date: End of date range
            chunk_size_days: Size of each chunk in days
            **kwargs: Additional arguments for API calls

        Returns:
            Summary of sync operation
        """
        chunks = split_into_chunks(start_date, end_date, chunk_size_days)

        # Generate job ID
        job_id = hashlib.md5(
            f"{metric_name}-{start_date.isoformat()}-{end_date.isoformat()}".encode()
        ).hexdigest()[:16]

        progress = SyncJobProgress(
            job_id=job_id,
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date,
            total_chunks=len(chunks),
        )

        logger.info(
            "Starting large range sync",
            job_id=job_id,
            metric_name=metric_name,
            total_chunks=len(chunks),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        for chunk_start, chunk_end in chunks:
            try:
                # Fetch this chunk
                result = await self.get_metric_time_series(
                    fulcra_fetch_func,
                    metric_name,
                    chunk_start,
                    chunk_end,
                    **kwargs,
                )

                # Validate we actually got data
                if result.records_fetched == 0 and not result.from_cache:
                    logger.warning(
                        "Chunk returned no new data from API",
                        job_id=job_id,
                        chunk_start=chunk_start.isoformat(),
                        chunk_end=chunk_end.isoformat(),
                    )

                # Verify database write succeeded if we fetched new data
                if result.records_fetched > 0:
                    verify_data = self.db.query_metrics(metric_name, chunk_start, chunk_end)
                    if len(verify_data) == 0:
                        logger.error(
                            "Database write verification failed - no records found after insert",
                            job_id=job_id,
                            metric_name=metric_name,
                            records_fetched=result.records_fetched,
                        )
                        progress.failed.append({
                            "start": chunk_start.isoformat(),
                            "end": chunk_end.isoformat(),
                            "error": "Database write verification failed",
                        })
                        continue

                progress.completed += 1
                progress.last_success = chunk_end

                logger.info(
                    "Chunk completed",
                    job_id=job_id,
                    progress=f"{progress.completed}/{progress.total_chunks}",
                    from_cache=result.from_cache,
                    records_fetched=result.records_fetched,
                    records_in_db=len(verify_data) if result.records_fetched > 0 else "n/a",
                )

                # Respect rate limits between chunks
                if progress.completed < progress.total_chunks:
                    await asyncio.sleep(RATE_LIMIT_DELAY)

            except RateLimitError as e:
                logger.warning(
                    "Rate limited during sync",
                    job_id=job_id,
                    retry_after=e.retry_after,
                )
                await asyncio.sleep(e.retry_after or 60)
                # Retry this chunk by decrementing the counter logic won't help here
                # Just continue and it will be a gap next time

            except AuthenticationError as e:
                logger.error(
                    "Authentication error during sync",
                    job_id=job_id,
                    error=str(e),
                )
                progress.status = "auth_failed"
                progress.failed.append(
                    {
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                        "error": "Authentication failed",
                    }
                )
                # Don't continue if auth failed
                break

            except Exception as e:
                logger.error(
                    "Chunk failed",
                    job_id=job_id,
                    chunk_start=chunk_start.isoformat(),
                    chunk_end=chunk_end.isoformat(),
                    error=str(e),
                )
                progress.failed.append(
                    {
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                        "error": str(e),
                    }
                )
                # Continue with next chunk

        # Final status
        if progress.completed == progress.total_chunks:
            progress.status = "completed"
        elif progress.status != "auth_failed":
            progress.status = "partial"

        return {
            "job_id": progress.job_id,
            "status": progress.status,
            "completed_chunks": progress.completed,
            "total_chunks": progress.total_chunks,
            "failed_chunks": len(progress.failed),
            "failed_details": progress.failed,
            "completion_rate": progress.completed / progress.total_chunks if progress.total_chunks > 0 else 0,
            "last_success": progress.last_success.isoformat() if progress.last_success else None,
        }


def format_sync_error(error: Exception, context: dict) -> str:
    """
    Convert technical errors into user-friendly messages.

    Args:
        error: The exception that occurred
        context: Context about the operation

    Returns:
        User-friendly error message
    """
    if isinstance(error, RateLimitError):
        return (
            f"Fulcra API rate limit reached. "
            f"Pausing for {error.retry_after}s then resuming automatically. "
            f"Progress saved: {context.get('completed', 0)}/{context.get('total', '?')} chunks."
        )

    if isinstance(error, AuthenticationError):
        return (
            f"Authentication expired. "
            f"Please re-authenticate with Fulcra. "
            f"Progress saved: {context.get('completed', 0)}/{context.get('total', '?')} chunks."
        )

    if isinstance(error, APITimeoutError):
        return (
            f"Request timed out. "
            f"This can happen with large date ranges. "
            f"Progress: {context.get('completed', 0)}/{context.get('total', '?')} chunks."
        )

    if isinstance(error, DataNotFoundError):
        return (
            f"No data available for the requested time range. "
            f"This might mean you didn't have a device recording data then."
        )

    return (
        f"Unexpected error: {str(error)} "
        f"Progress saved: {context.get('completed', 0)}/{context.get('total', '?')} chunks. "
        f"You can resume this sync later with: health_db_resume_sync('{context.get('job_id', 'unknown')}')"
    )


# Module-level singleton for easy access
_fetcher_instance: SmartFetcher | None = None


def get_smart_fetcher() -> SmartFetcher:
    """Get the global smart fetcher instance."""
    global _fetcher_instance

    if _fetcher_instance is None:
        _fetcher_instance = SmartFetcher()

    return _fetcher_instance


def reset_smart_fetcher():
    """Reset the global smart fetcher instance (for testing)."""
    global _fetcher_instance
    _fetcher_instance = None
