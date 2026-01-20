#!/usr/bin/env python3
"""
Direct test of metric fetching to debug the empty data issue.
"""
import asyncio
import sys
from datetime import datetime, timezone

# Add the module to path
sys.path.insert(0, '/Users/paulregan/GitRepo/fulcra-context-mcp')

from fulcra_mcp.smart_fetch import SmartFetcher
from fulcra_mcp.health_db import HealthDatabase
from fulcra_context import fulcra  # The actual Fulcra API client


async def test_metric_fetch():
    """Test metric fetching directly."""
    print("=" * 80)
    print("Testing metric fetch with debug output")
    print("=" * 80)

    # Initialize database and smart fetcher
    db = HealthDatabase()
    smart_fetcher = SmartFetcher(db)

    # Check database stats
    stats = db.stats()
    print(f"\nDatabase stats BEFORE:")
    print(f"  health_metrics_count: {stats['health_metrics_count']}")
    print(f"  database_size_mb: {stats['database_size_mb']}")

    # Test time range
    start_time = datetime(2026, 1, 13, 6, 19, 6, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 13, 6, 43, 34, tzinfo=timezone.utc)

    print(f"\nFetching heart_rate from {start_time} to {end_time}")

    # Check for gaps
    gaps = db.identify_gaps("heart_rate", start_time, end_time)
    print(f"\nGaps identified: {len(gaps)}")
    for gap in gaps:
        print(f"  Gap: {gap.start_time} to {gap.end_time}")

    # Fetch using smart fetcher
    print("\nCalling smart_fetcher.get_metric_time_series()...")
    try:
        result = await smart_fetcher.get_metric_time_series(
            fulcra_fetch_func=fulcra.metric_time_series,
            metric_name="heart_rate",
            start_time=start_time,
            end_time=end_time,
            sample_rate=60,
        )

        print(f"\nResult:")
        print(f"  from_cache: {result.from_cache}")
        print(f"  gaps_fetched: {result.gaps_fetched}")
        print(f"  records_fetched: {result.records_fetched}")
        print(f"  records_from_cache: {result.records_from_cache}")
        print(f"  data length: {len(result.data)}")

        if result.data:
            print(f"  First record: {result.data[0]}")
            print(f"  Last record: {result.data[-1]}")
        else:
            print(f"  DATA IS EMPTY!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    # Check database stats again
    stats = db.stats()
    print(f"\nDatabase stats AFTER:")
    print(f"  health_metrics_count: {stats['health_metrics_count']}")
    print(f"  database_size_mb: {stats['database_size_mb']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_metric_fetch())
