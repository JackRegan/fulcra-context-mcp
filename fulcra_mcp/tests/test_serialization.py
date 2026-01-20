"""
Data Normalization & Serialization Tests

Tests for JSON serialization of complex data types from Fulcra API:
- Numpy arrays and scalars
- Datetime objects and timezone handling
- Field name mapping (Fulcra API → database schema)
- Complex nested JSON structures
- None/null value handling
"""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from fulcra_mcp.health_db import HealthDatabase
from fulcra_mcp.smart_fetch import (
    SmartFetcher,
    _convert_timestamps_in_records,
    _normalize_metric_records,
    _normalize_workout_records,
    _normalize_sleep_records
)


class TestNumpyConversion:
    """Test numpy array and scalar conversion to Python types."""

    def test_numpy_scalar_to_python(self, clean_db: HealthDatabase):
        """Test that numpy scalars (float64, int64) convert to Python types."""
        # Simulate Fulcra API response with numpy scalars
        numpy_value = np.float64(72.5)

        # Convert using the smart fetcher's conversion method
        fetcher = SmartFetcher(db=clean_db)
        records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "value": numpy_value,
                "metadata": {"source": "test"}
            }
        ]

        # Normalize records
        normalized = _convert_timestamps_in_records(records)

        # Verify conversion
        assert isinstance(normalized[0]["value"], (int, float))
        assert not isinstance(normalized[0]["value"], np.generic)

        # Verify JSON serialization works
        json_str = json.dumps(normalized)
        assert json_str is not None

        # Verify deserialization
        deserialized = json.loads(json_str)
        assert deserialized[0]["value"] == 72.5

    def test_numpy_array_to_list(self, clean_db: HealthDatabase):
        """Test that numpy arrays convert to Python lists."""
        # Simulate API response with numpy array
        numpy_array = np.array([1.0, 2.0, 3.0, 4.0])

        records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "values": numpy_array,
                "metadata": {"source": "test"}
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _convert_timestamps_in_records(records)

        # Verify conversion
        assert isinstance(normalized[0]["values"], list)
        assert not isinstance(normalized[0]["values"], np.ndarray)

        # Verify JSON serialization
        json_str = json.dumps(normalized)
        deserialized = json.loads(json_str)
        assert deserialized[0]["values"] == [1.0, 2.0, 3.0, 4.0]

    def test_nested_numpy_values(self, clean_db: HealthDatabase):
        """Test numpy values nested in complex structures."""
        records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "workout_metrics": {
                    "avg_heart_rate": np.float64(145.5),
                    "max_heart_rate": np.int64(178),
                    "heart_rate_samples": np.array([140, 145, 150, 155])
                }
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _convert_timestamps_in_records(records)

        # Verify nested conversions
        metrics = normalized[0]["workout_metrics"]
        assert isinstance(metrics["avg_heart_rate"], float)
        assert isinstance(metrics["max_heart_rate"], int)
        assert isinstance(metrics["heart_rate_samples"], list)

        # Verify JSON serialization
        json_str = json.dumps(normalized)
        assert json_str is not None


class TestTimestampHandling:
    """Test timestamp conversion and serialization."""

    def test_iso8601_string_to_timestamp(self, clean_db: HealthDatabase):
        """Test ISO8601 string conversion to Unix timestamp."""
        iso_string = "2026-01-18T12:00:00Z"

        timestamp = clean_db.to_timestamp(iso_string)

        # Verify it's a float
        assert isinstance(timestamp, float)

        # Verify round-trip conversion
        dt = clean_db.from_timestamp(timestamp)
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 18
        assert dt.hour == 12

    def test_datetime_object_to_timestamp(self, clean_db: HealthDatabase):
        """Test datetime object conversion to Unix timestamp."""
        dt = datetime(2026, 1, 18, 12, 0, 0, tzinfo=timezone.utc)

        timestamp = clean_db.to_timestamp(dt)

        # Verify it's a float
        assert isinstance(timestamp, float)

        # Verify round-trip
        dt_back = clean_db.from_timestamp(timestamp)
        assert dt_back == dt

    def test_timestamp_float_passthrough(self, clean_db: HealthDatabase):
        """Test that Unix timestamps are passed through."""
        unix_timestamp = 1737201600.0  # 2026-01-18 12:00:00 UTC

        result = clean_db.to_timestamp(unix_timestamp)

        assert result == unix_timestamp

    def test_timezone_aware_datetime(self, clean_db: HealthDatabase):
        """Test timezone-aware datetime handling."""
        # Create datetime with timezone
        dt_utc = datetime(2026, 1, 18, 12, 0, 0, tzinfo=timezone.utc)

        timestamp = clean_db.to_timestamp(dt_utc)
        dt_back = clean_db.from_timestamp(timestamp)

        # Should always return UTC datetime
        assert dt_back.tzinfo == timezone.utc

    def test_timestamp_in_json_response(self, clean_db: HealthDatabase):
        """Test that timestamps in JSON responses are serializable."""
        records = [
            {
                "start_date": datetime(2026, 1, 18, 12, 0, 0, tzinfo=timezone.utc),
                "end_date": "2026-01-18T12:01:00Z",
                "value": 72.5
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _convert_timestamps_in_records(records)

        # Verify JSON serialization works
        json_str = json.dumps(normalized, default=str)
        assert json_str is not None

        # Timestamps should be strings in JSON
        deserialized = json.loads(json_str)
        assert isinstance(deserialized[0]["start_date"], str)


class TestFieldNameMapping:
    """Test Fulcra API field name mapping to database schema."""

    def test_metric_field_mapping(self, clean_db: HealthDatabase):
        """Test that start_date maps to timestamp for metrics."""
        # Fulcra API format
        api_records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "end_date": "2026-01-18T12:01:00Z",
                "value": 72.5
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _normalize_metric_records(api_records)

        # Should have 'timestamp' field for database
        assert "timestamp" in normalized[0]
        assert isinstance(normalized[0]["timestamp"], float)

    def test_workout_field_preservation(self, clean_db: HealthDatabase):
        """Test that workout fields are preserved correctly."""
        api_records = [
            {
                "id": "workout_123",
                "start_date": "2026-01-18T07:00:00Z",
                "end_date": "2026-01-18T08:00:00Z",
                "workout_activity_type": "running"
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _normalize_workout_records(api_records)

        # Should have start_time/end_time as floats
        assert "start_time" in normalized[0]
        assert "end_time" in normalized[0]
        assert isinstance(normalized[0]["start_time"], float)
        assert isinstance(normalized[0]["end_time"], float)

    def test_none_value_handling(self, clean_db: HealthDatabase):
        """Test that None/null values are handled gracefully."""
        api_records = [
            {
                "start_date": "2026-01-18T12:00:00Z",
                "value": None,
                "metadata": None
            }
        ]

        fetcher = SmartFetcher(db=clean_db)
        normalized = _normalize_metric_records(api_records)

        # None values should be preserved
        assert normalized[0]["value"] is None

        # Should be JSON serializable
        json_str = json.dumps(normalized)
        assert json_str is not None


class TestComplexNestedJSON:
    """Test complex nested JSON structures."""

    def test_workout_nested_json(self, sample_workout_data, clean_db: HealthDatabase):
        """Test workout data with nested metrics."""
        fetcher = SmartFetcher(db=clean_db)

        # Normalize workout data
        normalized = _normalize_workout_records(sample_workout_data)

        # Verify JSON serialization
        json_str = json.dumps(normalized, default=str)
        assert json_str is not None

        # Verify nested structure preserved
        deserialized = json.loads(json_str)
        assert "metrics" in deserialized[0]
        assert deserialized[0]["metrics"]["avg_heart_rate"] == 145

    def test_sleep_nested_json(self, sample_sleep_data, clean_db: HealthDatabase):
        """Test sleep cycle data with nested stages."""
        fetcher = SmartFetcher(db=clean_db)

        # Normalize sleep data
        normalized = _normalize_sleep_records(sample_sleep_data)

        # Verify JSON serialization
        json_str = json.dumps(normalized, default=str)
        assert json_str is not None

        # Verify nested sleep stages preserved
        deserialized = json.loads(json_str)
        assert "sleep_stages" in deserialized[0]
        assert len(deserialized[0]["sleep_stages"]) == 4

    def test_database_json_round_trip(self, sample_workout_data, clean_db: HealthDatabase):
        """Test that complex JSON survives database storage and retrieval."""
        fetcher = SmartFetcher(db=clean_db)

        # Normalize and insert
        normalized = _normalize_workout_records(sample_workout_data)
        clean_db.insert_workouts(normalized)

        # Query back
        workouts = clean_db.query_workouts(
            start_time=clean_db.to_timestamp("2026-01-18T00:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-19T00:00:00Z")
        )

        # Verify nested structure preserved
        # Note: query_workouts returns already-parsed workout objects
        assert len(workouts) == 1
        assert "metrics" in workouts[0]
        assert workouts[0]["metrics"]["avg_heart_rate"] == 145


class TestEndToEndSerialization:
    """Test complete serialization flow: API → Database → Query → JSON."""

    def test_metrics_serialization_flow(self, sample_metrics_data, clean_db: HealthDatabase):
        """Test complete flow for metrics data."""
        fetcher = SmartFetcher(db=clean_db)

        # Step 1: Convert timestamps and numpy types
        converted = _convert_timestamps_in_records(sample_metrics_data)

        # Step 2: Normalize API data
        normalized = _normalize_metric_records(converted)

        # Step 3: Store in database
        clean_db.insert_metrics("heart_rate", normalized)

        # Step 3: Query from database
        queried = clean_db.query_metrics(
            "heart_rate",
            start_time=clean_db.to_timestamp("2026-01-18T11:00:00Z"),
            end_time=clean_db.to_timestamp("2026-01-18T13:00:00Z")
        )

        # Step 4: Verify JSON serialization
        json_str = json.dumps(queried, default=str)
        assert json_str is not None

        # Step 5: Verify data integrity
        deserialized = json.loads(json_str)
        assert len(deserialized) == 3
        assert all("timestamp" in record for record in deserialized)
        assert all("value" in record for record in deserialized)

    def test_full_response_serialization(self, sample_metrics_data, clean_db: HealthDatabase):
        """Test serialization of a complete MCP tool response."""
        fetcher = SmartFetcher(db=clean_db)

        # Simulate MCP tool response format
        # Step 1: Convert timestamps and numpy types
        converted = _convert_timestamps_in_records(sample_metrics_data)

        # Step 2: Normalize and store
        normalized = _normalize_metric_records(converted)
        clean_db.insert_metrics("heart_rate", normalized)

        response = {
            "metric_name": "heart_rate",
            "start_time": "2026-01-18T11:00:00Z",
            "end_time": "2026-01-18T13:00:00Z",
            "records": normalized,
            "cache_info": {
                "from_cache": True,
                "records_fetched": 0,
                "records_from_cache": 3
            }
        }

        # This is what gets sent to the MCP client - must be JSON serializable
        json_str = json.dumps(response, default=str)
        assert json_str is not None

        # Verify deserialization
        deserialized = json.loads(json_str)
        assert deserialized["metric_name"] == "heart_rate"
        assert len(deserialized["records"]) == 3
