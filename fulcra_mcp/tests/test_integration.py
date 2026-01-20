"""
Database Integration Tests

Integration tests against real Fulcra API to validate:
- Database CRUD operations for all data types
- Smart fetch gap detection
- Metadata caching
- Export functionality
- Database management operations
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fulcra_mcp.health_db import HealthDatabase
from fulcra_mcp.smart_fetch import (
    SmartFetcher,
    _normalize_metric_records,
    _normalize_workout_records,
    _normalize_sleep_records
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestMetricTimeSeriesCRUD:
    """Test metric time series database operations."""

    @pytest.mark.slow
    async def test_fetch_and_store_heart_rate(self, clean_db: HealthDatabase, fulcra_client):
        """Test fetching heart_rate data and storing in database."""
        # Note: This requires real Fulcra API access
        # Skip if API not available
        pytest.skip("Requires manual Fulcra API integration - implement when client is available")

        # TODO: Implement when Fulcra API client is integrated
        # fetcher = SmartFetcher(db=clean_db, client=fulcra_client)
        #
        # end_time = datetime.now(timezone.utc)
        # start_time = end_time - timedelta(days=7)
        #
        # result = await fetcher.get_metric_time_series(
        #     metric_name="heart_rate",
        #     start_time=start_time.isoformat(),
        #     end_time=end_time.isoformat()
        # )
        #
        # # Verify data stored
        # assert result.records_fetched > 0
        #
        # # Query database directly
        # metrics = clean_db.query_metrics(
        #     "heart_rate",
        #     start_time=clean_db.to_timestamp(start_time),
        #     end_time=clean_db.to_timestamp(end_time)
        # )
        #
        # assert len(metrics) > 0
        # assert all("timestamp" in m for m in metrics)
        # assert all("value" in m for m in metrics)

    def test_metric_insert_and_query(self, clean_db: HealthDatabase):
        """Test direct metric insert and query operations."""
        # Insert test data
        test_metrics = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "value": 72.5,
                "metadata": {"source": "test"}
            },
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:01:00Z"),
                "value": 73.0,
                "metadata": {"source": "test"}
            },
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:02:00Z"),
                "value": 74.2,
                "metadata": {"source": "test"}
            }
        ]

        clean_db.insert_metrics("heart_rate", test_metrics)

        # Query back
        queried = clean_db.query_metrics(
            "heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T11:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-18T13:00:00Z")
        )

        # Verify
        assert len(queried) == 3
        assert queried[0]["value"] == 72.5
        assert queried[1]["value"] == 73.0
        assert queried[2]["value"] == 74.2

    def test_metric_upsert(self, clean_db: HealthDatabase):
        """Test that upsert updates existing records."""
        # Insert initial data
        initial = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "value": 72.5
            }
        ]
        clean_db.insert_metrics("heart_rate", initial)

        # Upsert with different value (simulating recent data re-fetch)
        updated = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "value": 75.0  # Updated value
            }
        ]
        clean_db.upsert_metrics("heart_rate", updated)

        # Query and verify update
        queried = clean_db.query_metrics(
            "heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T11:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-18T13:00:00Z")
        )

        assert len(queried) == 1
        assert queried[0]["value"] == 75.0

    def test_json_serialization_of_query_results(self, clean_db: HealthDatabase):
        """Test that query results are JSON serializable."""
        # Insert data
        test_metrics = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "value": 72.5,
                "value_json": json.dumps({"min": 70, "max": 75}),
                "metadata": json.dumps({"source": "apple_watch"})
            }
        ]
        clean_db.insert_metrics("heart_rate", test_metrics)

        # Query
        queried = clean_db.query_metrics(
            "heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T11:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-18T13:00:00Z")
        )

        # Verify JSON serialization works
        json_str = json.dumps(queried, default=str)
        assert json_str is not None

        # Verify deserialization
        deserialized = json.loads(json_str)
        assert len(deserialized) == 1


class TestWorkoutsStorage:
    """Test workouts database operations."""

    def test_insert_and_query_workouts(self, clean_db: HealthDatabase, sample_workout_data):
        """Test workout storage and retrieval."""
        # Prepare workout data
        workout_records = []
        for workout in sample_workout_data:
            workout_records.append({
                "workout_id": workout["id"],
                "start_time": clean_db.to_timestamp(workout["start_date"]),
                "end_time": clean_db.to_timestamp(workout["end_date"]),
                "data_json": json.dumps(workout)
            })

        # Insert
        clean_db.insert_workouts(workout_records)

        # Query
        workouts = clean_db.query_workouts(
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z")
        )

        # Verify
        assert len(workouts) == 1
        assert workouts[0]["workout_id"] == "workout_123"

        # Verify nested JSON preserved
        workout_data = json.loads(workouts[0]["data_json"])
        assert workout_data["workout_activity_type"] == "running"
        assert workout_data["metrics"]["avg_heart_rate"] == 145

    def test_workout_deduplication(self, clean_db: HealthDatabase, sample_workout_data):
        """Test that duplicate workouts are not inserted."""
        # Prepare workout
        workout_records = [{
            "workout_id": sample_workout_data[0]["id"],
            "start_time": clean_db.to_timestamp(sample_workout_data[0]["start_date"]),
            "end_time": clean_db.to_timestamp(sample_workout_data[0]["end_date"]),
            "data_json": json.dumps(sample_workout_data[0])
        }]

        # Insert twice
        clean_db.insert_workouts(workout_records)
        clean_db.insert_workouts(workout_records)  # Should be ignored

        # Query
        workouts = clean_db.query_workouts(
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z")
        )

        # Should only have one workout
        assert len(workouts) == 1

    def test_has_workouts_for_range(self, clean_db: HealthDatabase, sample_workout_data):
        """Test the has_workouts_for_range helper."""
        # Insert workout
        workout_records = [{
            "workout_id": sample_workout_data[0]["id"],
            "start_time": clean_db.to_timestamp(sample_workout_data[0]["start_date"]),
            "end_time": clean_db.to_timestamp(sample_workout_data[0]["end_date"]),
            "data_json": json.dumps(sample_workout_data[0])
        }]
        clean_db.insert_workouts(workout_records)

        # Check range with workout
        has_workouts = clean_db.has_workouts_for_range(
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z")
        )
        assert has_workouts is True

        # Check range without workout
        has_workouts = clean_db.has_workouts_for_range(
            start_time=clean_db.to_timestamp("2026-01-17T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-17T23:59:59Z")
        )
        assert has_workouts is False


class TestSleepCyclesStorage:
    """Test sleep cycles database operations."""

    def test_insert_and_query_sleep(self, clean_db: HealthDatabase, sample_sleep_data):
        """Test sleep cycle storage and retrieval."""
        # Prepare sleep data
        sleep_records = []
        for sleep in sample_sleep_data:
            sleep_records.append({
                "cycle_id": sleep["id"],
                "start_time": clean_db.to_timestamp(sleep["start_date"]),
                "end_time": clean_db.to_timestamp(sleep["end_date"]),
                "data_json": json.dumps(sleep)
            })

        # Insert
        clean_db.insert_sleep_cycles(sleep_records)

        # Query
        cycles = clean_db.query_sleep_cycles(
            start_time=clean_db.to_timestamp("2026-01-17T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z")
        )

        # Verify
        assert len(cycles) == 1
        assert cycles[0]["cycle_id"] == "sleep_456"

        # Verify sleep stages preserved
        sleep_data = json.loads(cycles[0]["data_json"])
        assert len(sleep_data["sleep_stages"]) == 4
        assert sleep_data["sleep_stages"][0]["stage"] == "awake"


class TestLocationData:
    """Test location data storage and retrieval."""

    def test_insert_and_query_locations(self, clean_db: HealthDatabase):
        """Test location data storage."""
        # Insert test locations
        locations = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "latitude": 37.7749,
                "longitude": -122.4194,
                "accuracy": 10.0,
                "data_json": json.dumps({"city": "San Francisco"})
            },
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:15:00Z"),
                "latitude": 37.7849,
                "longitude": -122.4094,
                "accuracy": 15.0,
                "data_json": json.dumps({"city": "San Francisco"})
            }
        ]

        clean_db.insert_locations(locations)

        # Query
        queried = clean_db.query_locations(
            start_time=clean_db.to_timestamp("2026-01-18T11:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-18T13:00:00Z")
        )

        # Verify
        assert len(queried) == 2
        assert queried[0]["latitude"] == 37.7749
        assert queried[0]["longitude"] == -122.4194

    def test_get_location_at_time(self, clean_db: HealthDatabase):
        """Test get_location_at_time with window search."""
        # Insert location
        locations = [{
            "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
            "latitude": 37.7749,
            "longitude": -122.4194,
            "accuracy": 10.0
        }]
        clean_db.insert_locations(locations)

        # Query exact time
        location = clean_db.get_location_at_time(
            time=clean_db.to_timestamp("2026-01-18T12:00:00Z")
        )
        assert location is not None
        assert location["latitude"] == 37.7749

        # Query within window (5 minutes later)
        location = clean_db.get_location_at_time(
            time=clean_db.to_timestamp("2026-01-18T12:05:00Z"),
            window_size=600  # 10 minute window
        )
        assert location is not None

        # Query outside window
        location = clean_db.get_location_at_time(
            time=clean_db.to_timestamp("2026-01-18T13:00:00Z"),
            window_size=600
        )
        assert location is None


class TestMetadataCaching:
    """Test metadata caching with TTL."""

    def test_set_and_get_metadata(self, clean_db: HealthDatabase):
        """Test basic metadata storage and retrieval."""
        # Set metadata
        user_info = {
            "user_id": "test_user",
            "timezone": "America/Los_Angeles"
        }
        clean_db.set_metadata("user_info", json.dumps(user_info))

        # Get metadata
        cached = clean_db.get_metadata("user_info")
        assert cached is not None

        data = json.loads(cached)
        assert data["user_id"] == "test_user"

    def test_metadata_expiry(self, clean_db: HealthDatabase):
        """Test that metadata expires after TTL."""
        # Set metadata with 1 second TTL
        clean_db.set_metadata("test_key", "test_value", ttl=1)

        # Should exist immediately
        cached = clean_db.get_metadata("test_key")
        assert cached == "test_value"

        # Wait for expiry
        time.sleep(2)

        # Should be None after expiry
        cached = clean_db.get_metadata("test_key")
        assert cached is None

    def test_metadata_max_age(self, clean_db: HealthDatabase):
        """Test max_age parameter."""
        # Set metadata
        clean_db.set_metadata("test_key", "test_value")

        # Should exist with large max_age
        cached = clean_db.get_metadata("test_key", max_age=3600)
        assert cached == "test_value"

        # Should be None with very small max_age
        cached = clean_db.get_metadata("test_key", max_age=0.001)
        time.sleep(0.01)
        cached = clean_db.get_metadata("test_key", max_age=0.001)
        assert cached is None


class TestDatabaseStats:
    """Test database statistics and info methods."""

    def test_get_stats_empty_db(self, clean_db: HealthDatabase):
        """Test stats on empty database."""
        stats = clean_db.get_stats()

        assert stats is not None
        assert stats["enabled"] is True
        assert stats["total_records"] == 0
        assert stats["records_by_type"]["health_metrics"] == 0

    def test_get_stats_with_data(self, clean_db: HealthDatabase):
        """Test stats with data."""
        # Insert test data
        test_metrics = [
            {
                "timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"),
                "value": 72.5
            }
        ]
        clean_db.insert_metrics("heart_rate", test_metrics)

        stats = clean_db.get_stats()

        # Verify counts
        assert stats["total_records"] == 1
        assert stats["records_by_type"]["health_metrics"] == 1

        # Verify metrics breakdown
        assert "metrics_breakdown" in stats
        assert any(m["metric_name"] == "heart_rate" for m in stats["metrics_breakdown"])

    def test_database_size(self, clean_db: HealthDatabase):
        """Test database size reporting."""
        stats = clean_db.get_stats()

        assert "database_size_mb" in stats
        assert isinstance(stats["database_size_mb"], (int, float))
        assert stats["database_size_mb"] >= 0


class TestDatabaseClearing:
    """Test selective database clearing."""

    def test_clear_all_data(self, clean_db: HealthDatabase):
        """Test clearing all data."""
        # Insert test data
        test_metrics = [
            {"timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"), "value": 72.5}
        ]
        clean_db.insert_metrics("heart_rate", test_metrics)

        # Verify data exists
        stats = clean_db.get_stats()
        assert stats["total_records"] == 1

        # Clear all
        clean_db.clear()

        # Verify empty
        stats = clean_db.get_stats()
        assert stats["total_records"] == 0

    def test_clear_old_data_only(self, clean_db: HealthDatabase):
        """Test clearing data older than N days."""
        # Insert old data (91 days ago)
        old_timestamp = clean_db.to_timestamp(
            datetime.now(timezone.utc) - timedelta(days=91)
        )
        old_metrics = [{"timestamp": old_timestamp, "value": 60.0}]
        clean_db.insert_metrics("heart_rate", old_metrics)

        # Insert recent data (1 day ago)
        recent_timestamp = clean_db.to_timestamp(
            datetime.now(timezone.utc) - timedelta(days=1)
        )
        recent_metrics = [{"timestamp": recent_timestamp, "value": 75.0}]
        clean_db.insert_metrics("heart_rate", recent_metrics)

        # Clear data older than 30 days
        clean_db.clear(older_than_days=30)

        # Query all data
        all_metrics = clean_db.query_metrics(
            "heart_rate",
            start_time=0,
            end_time=time.time()
        )

        # Should only have recent data
        assert len(all_metrics) == 1
        assert all_metrics[0]["value"] == 75.0


class TestExportFunctionality:
    """Test data export to CSV and JSON."""

    def test_csv_export(self, clean_db: HealthDatabase, test_exports_path: Path):
        """Test CSV export of metrics."""
        # Insert test data
        test_metrics = [
            {"timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"), "value": 72.5},
            {"timestamp": clean_db.to_timestamp("2026-01-18T12:01:00Z"), "value": 73.0},
        ]
        clean_db.insert_metrics("heart_rate", test_metrics)

        # Export to CSV
        export_path = clean_db.export_metrics_csv(
            metric_name="heart_rate",
            output_dir=test_exports_path
        )

        # Verify file exists
        assert export_path.exists()
        assert export_path.suffix == ".csv"

        # Verify content
        content = export_path.read_text()
        assert "timestamp" in content
        assert "value" in content
        assert "72.5" in content

    def test_json_export(self, clean_db: HealthDatabase, test_exports_path: Path):
        """Test JSON export of all data."""
        # Insert various data types
        clean_db.insert_metrics("heart_rate", [
            {"timestamp": clean_db.to_timestamp("2026-01-18T12:00:00Z"), "value": 72.5}
        ])

        # Export to JSON
        export_path = clean_db.export_json(output_dir=test_exports_path)

        # Verify file exists
        assert export_path.exists()
        assert export_path.suffix == ".json"

        # Verify content
        content = json.loads(export_path.read_text())
        assert "health_metrics" in content
        assert len(content["health_metrics"]) == 1


class TestSmartFetchGapDetection:
    """Test smart fetch gap detection scenarios."""

    def test_empty_database_cold_start(self, clean_db: HealthDatabase):
        """Test gap detection on empty database (cold start)."""
        # Query with empty database
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z"),
            expected_interval=60
        )

        # Should identify entire range as a gap
        assert len(gaps) == 1
        assert gaps[0].start_time == clean_db.to_timestamp("2026-01-18T00:00:00Z")
        assert gaps[0].end_time == clean_db.to_timestamp("2026-01-19T00:00:00Z")

    def test_partial_data_gap_filling(self, clean_db: HealthDatabase):
        """Test gap detection with partial data (gap filling scenario)."""
        # Insert data for days 1-3
        for hour in range(0, 72):  # 3 days
            timestamp = clean_db.to_timestamp(
                datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=hour)
            )
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0 + hour
            }])

        # Insert data for days 6-7 (skip days 4-5)
        for hour in range(120, 168):  # Days 6-7
            timestamp = clean_db.to_timestamp(
                datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=hour)
            )
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0 + hour
            }])

        # Query full week
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-25T00:00:00Z"),
            expected_interval=3600  # 1 hour
        )

        # Should identify gap for days 4-5
        assert len(gaps) > 0

        # Find the main gap (days 4-5)
        main_gap = max(gaps, key=lambda g: g.duration_hours)
        assert main_gap.duration_hours >= 48  # At least 2 days

    def test_no_gaps_fully_cached(self, clean_db: HealthDatabase):
        """Test gap detection when data is fully cached."""
        # Insert complete data for 24 hours
        start_time = datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc)
        for minute in range(0, 1440):  # 24 hours in minutes
            timestamp = clean_db.to_timestamp(start_time + timedelta(minutes=minute))
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0 + (minute % 30)
            }])

        # Query same range
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp(start_time),
            end_time=clean_db.to_timestamp(start_time + timedelta(hours=24)),
            expected_interval=60
        )

        # Should have no gaps (or only very small ones due to rounding)
        large_gaps = [g for g in gaps if g.duration_hours > 0.1]
        assert len(large_gaps) == 0

    def test_recent_data_window(self, clean_db: HealthDatabase):
        """Test that recent data (last 2 hours) is treated specially."""
        from fulcra_mcp.health_db import HealthDatabase

        # Get current time
        now = datetime.now(timezone.utc)

        # Insert data from 3 hours ago to 1 hour ago
        for minutes_ago in range(180, 60, -1):
            timestamp = clean_db.to_timestamp(now - timedelta(minutes=minutes_ago))
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0
            }])

        # Query including recent data (last 3 hours)
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp(now - timedelta(hours=3)),
            end_time=clean_db.to_timestamp(now),
            expected_interval=60
        )

        # Should identify gap in last hour
        recent_gaps = [g for g in gaps if g.end_time >= (now - timedelta(hours=1)).timestamp()]
        assert len(recent_gaps) > 0

    def test_gap_at_start_of_range(self, clean_db: HealthDatabase):
        """Test gap detection at the start of requested range."""
        # Insert data starting from hour 5
        start_time = datetime(2026, 1, 18, 5, 0, 0, tzinfo=timezone.utc)
        for hour in range(0, 5):
            timestamp = clean_db.to_timestamp(start_time + timedelta(hours=hour))
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0
            }])

        # Query from hour 0
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp(datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc)),
            end_time=clean_db.to_timestamp(datetime(2026, 1, 18, 10, 0, 0, tzinfo=timezone.utc)),
            expected_interval=3600
        )

        # Should identify gap at start (hours 0-5)
        assert len(gaps) > 0
        first_gap = gaps[0]
        assert first_gap.start_time == clean_db.to_timestamp(
            datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc)
        )

    def test_gap_at_end_of_range(self, clean_db: HealthDatabase):
        """Test gap detection at the end of requested range."""
        # Insert data up to hour 20
        start_time = datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc)
        for hour in range(0, 20):
            timestamp = clean_db.to_timestamp(start_time + timedelta(hours=hour))
            clean_db.insert_metrics("heart_rate", [{
                "timestamp": timestamp,
                "value": 70.0
            }])

        # Query to hour 24
        gaps = clean_db.identify_gaps(
            metric_name="heart_rate",
            start_time=clean_db.to_timestamp(start_time),
            end_time=clean_db.to_timestamp(start_time + timedelta(hours=24)),
            expected_interval=3600
        )

        # Should identify gap at end (hours 20-24)
        assert len(gaps) > 0
        last_gap = gaps[-1]
        assert last_gap.end_time == clean_db.to_timestamp(start_time + timedelta(hours=24))


class TestSmartFetcherIntegration:
    """Test SmartFetcher integration with database."""

    def test_normalize_metric_records(self, clean_db: HealthDatabase):
        """Test metric record normalization."""
        fetcher = SmartFetcher(db=clean_db)

        # API-style records
        api_records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "end_date": "2026-01-18T12:01:00Z",
                "value": 72.5
            }
        ]

        normalized = _normalize_metric_records(api_records)

        # Should have timestamp field
        assert "timestamp" in normalized[0]
        assert isinstance(normalized[0]["timestamp"], float)

    def test_normalize_workout_records(self, clean_db: HealthDatabase, sample_workout_data):
        """Test workout record normalization."""
        fetcher = SmartFetcher(db=clean_db)

        normalized = _normalize_workout_records(sample_workout_data)

        # Should have start_time/end_time as floats
        assert "start_time" in normalized[0]
        assert "end_time" in normalized[0]
        assert isinstance(normalized[0]["start_time"], float)

    def test_normalize_sleep_records(self, clean_db: HealthDatabase, sample_sleep_data):
        """Test sleep record normalization."""
        fetcher = SmartFetcher(db=clean_db)

        normalized = _normalize_sleep_records(sample_sleep_data)

        # Should have start_time/end_time as floats
        assert "start_time" in normalized[0]
        assert "end_time" in normalized[0]
        assert isinstance(normalized[0]["start_time"], float)


# Error handling and export tests in Phase 5
