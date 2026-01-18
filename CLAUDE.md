# Cache Implementation Plan v2.0
## "Personal Health Data Warehouse" Strategy

## Overview
Build a local SQLite database that acts as a personal health data warehouse, not a temporary cache. Once data is fetched from Fulcra, store it permanently for fast queries and offline access.

## Core Principles

1. **Immutable Data Philosophy**: Health data doesn't change once recorded
2. **Fetch Once, Store Forever**: Minimize API calls by treating SQLite as permanent storage
3. **Smart Sync**: Only fetch data that doesn't exist locally
4. **Recent Data Exception**: Very recent data (last 2 hours) may still be updating
5. **Reusable Archive**: SQLite db can be used by other tools (analysis, export, dashboards)

## Database Design

### Schema: Not a Cache, but a Health Database

```sql
-- Core health metrics table
CREATE TABLE health_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    timestamp REAL NOT NULL,          -- Unix timestamp
    value REAL,
    value_json TEXT,                  -- For complex values
    source TEXT,                      -- 'fulcra_api'
    fetched_at REAL NOT NULL,
    metadata TEXT,                    -- Additional context
    UNIQUE(metric_name, timestamp)    -- Prevent duplicates
);

CREATE INDEX idx_metric_time ON health_metrics(metric_name, timestamp);
CREATE INDEX idx_timestamp ON health_metrics(timestamp);

-- Workouts table
CREATE TABLE workouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workout_id TEXT UNIQUE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    workout_type TEXT,
    data_json TEXT NOT NULL,          -- Full workout data
    fetched_at REAL NOT NULL
);

CREATE INDEX idx_workout_time ON workouts(start_time, end_time);

-- Sleep cycles table
CREATE TABLE sleep_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT UNIQUE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    data_json TEXT NOT NULL,          -- Full sleep data
    fetched_at REAL NOT NULL
);

CREATE INDEX idx_sleep_time ON sleep_cycles(start_time, end_time);

-- Location data table
CREATE TABLE locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    latitude REAL,
    longitude REAL,
    accuracy REAL,
    data_json TEXT,
    fetched_at REAL NOT NULL,
    UNIQUE(timestamp)
);

CREATE INDEX idx_location_time ON locations(timestamp);

-- Metadata cache (for rarely-changing data)
CREATE TABLE metadata_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    expires_at REAL                   -- Only for truly dynamic data
);

-- Query log (track what we've fetched)
CREATE TABLE fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    query_params TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    fetched_at REAL NOT NULL,
    record_count INTEGER
);
```

## Smart Fetch Strategy

### Decision Tree: "Do we need to call the API?"

```python
async def get_metric_time_series(metric_name, start_time, end_time):
    """
    Smart fetch: Check local database first, only fetch gaps from API.
    """
    
    # 1. Check if we have this data locally
    local_data = db.query_time_range(metric_name, start_time, end_time)
    
    # 2. Identify gaps in local data
    gaps = identify_gaps(local_data, start_time, end_time)
    
    # 3. Only fetch missing data from API
    if gaps:
        for gap_start, gap_end in gaps:
            # Exception: If gap is in the last 2 hours, always re-fetch
            # (data might still be updating)
            if gap_end > now() - timedelta(hours=2):
                api_data = await fulcra.fetch(metric_name, gap_start, gap_end)
                db.upsert(api_data)  # Update or insert
            else:
                # Historical gap - fetch once and store forever
                api_data = await fulcra.fetch(metric_name, gap_start, gap_end)
                db.insert(api_data)  # Insert only (won't change)
    
    # 4. Return combined local data
    return db.query_time_range(metric_name, start_time, end_time)
```

### Gap Detection Logic

```python
def identify_gaps(local_data, requested_start, requested_end):
    """
    Find time ranges we don't have locally.
    
    Returns: List of (start, end) tuples representing gaps
    """
    if not local_data:
        # No local data - need entire range
        return [(requested_start, requested_end)]
    
    gaps = []
    
    # Check if we're missing data before first local record
    if local_data[0].timestamp > requested_start:
        gaps.append((requested_start, local_data[0].timestamp))
    
    # Check for gaps between local records
    for i in range(len(local_data) - 1):
        gap_size = local_data[i+1].timestamp - local_data[i].timestamp
        if gap_size > expected_interval:
            gaps.append((local_data[i].timestamp, local_data[i+1].timestamp))
    
    # Check if we're missing data after last local record
    if local_data[-1].timestamp < requested_end:
        gaps.append((local_data[-1].timestamp, requested_end))
    
    return gaps
```

## Error Handling Strategy

### Challenge: Large Requests & API Limits

When syncing large date ranges (e.g., 6 months of data), we face several risks:
- **Rate limiting**: Fulcra API throttles too many requests
- **Authentication expiry**: OAuth tokens expire during long operations
- **Network timeouts**: Large requests fail mid-transfer
- **Partial failures**: Some months succeed, others fail

### Solution: Robust Error Handling with Resume Capability

#### 1. Chunked Requests with Progress Tracking

**Break large requests into manageable chunks:**

```python
async def sync_large_range(
    metric_name: str,
    start_date: datetime,
    end_date: datetime,
    chunk_size_days: int = 7  # Fetch 1 week at a time
):
    """
    Sync large date range with progress tracking and error recovery.
    """
    chunks = split_into_chunks(start_date, end_date, chunk_size_days)
    
    progress = {
        "total_chunks": len(chunks),
        "completed": 0,
        "failed": [],
        "last_success": None
    }
    
    for chunk_start, chunk_end in chunks:
        try:
            # Fetch chunk
            data = await fetch_with_retry(
                metric_name, 
                chunk_start, 
                chunk_end,
                max_retries=3
            )
            
            # Store in database
            db.insert(data)
            
            # Update progress
            progress["completed"] += 1
            progress["last_success"] = chunk_end
            
            # Log progress
            logger.info(
                f"Progress: {progress['completed']}/{progress['total_chunks']} chunks"
            )
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)
            
        except RateLimitError as e:
            # Hit rate limit - pause and retry
            logger.warning(f"Rate limited, waiting {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            # Retry this chunk (don't increment)
            continue
            
        except AuthenticationError as e:
            # Auth token expired - re-authenticate
            logger.warning("Authentication expired, re-authenticating")
            await fulcra.authorize()
            # Retry this chunk
            continue
            
        except Exception as e:
            # Other error - log and continue with next chunk
            logger.error(f"Failed chunk {chunk_start} to {chunk_end}: {e}")
            progress["failed"].append({
                "start": chunk_start,
                "end": chunk_end,
                "error": str(e)
            })
            # Continue with next chunk (partial success is OK)
            continue
    
    # Return summary
    return {
        "success": progress["completed"],
        "failed": len(progress["failed"]),
        "failed_chunks": progress["failed"],
        "completion_rate": progress["completed"] / progress["total_chunks"]
    }
```

#### 2. Exponential Backoff with Retry Logic

```python
async def fetch_with_retry(
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict:
    """
    Fetch data with exponential backoff retry logic.
    """
    for attempt in range(max_retries):
        try:
            return await fulcra.fetch(metric_name, start_time, end_time)
            
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Final attempt failed
            
            # Exponential backoff: 1s, 2s, 4s, 8s...
            delay = base_delay * (2 ** attempt)
            
            # Use retry_after from API if provided
            if hasattr(e, 'retry_after'):
                delay = max(delay, e.retry_after)
            
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {delay}s"
            )
            await asyncio.sleep(delay)
            
        except AuthenticationError as e:
            # Re-authenticate and retry immediately
            logger.warning("Auth failed, re-authenticating")
            await fulcra.authorize()
            # Don't count this against retry limit
            continue
            
        except (TimeoutError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Network error, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
            
        except Exception as e:
            # Unknown error - don't retry
            logger.error(f"Unrecoverable error: {e}")
            raise
```

#### 3. Progress Persistence & Resume Support

**Store sync progress in database:**

```sql
CREATE TABLE sync_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    metric_name TEXT NOT NULL,
    start_date REAL NOT NULL,
    end_date REAL NOT NULL,
    status TEXT NOT NULL,              -- 'running', 'completed', 'failed', 'paused'
    progress_percent REAL,
    last_synced_date REAL,
    chunks_completed INTEGER,
    chunks_total INTEGER,
    chunks_failed TEXT,                -- JSON array of failed chunks
    started_at REAL NOT NULL,
    completed_at REAL,
    error_message TEXT
);
```

**Resume interrupted syncs:**

```python
async def resume_sync(job_id: str):
    """
    Resume a previously interrupted sync job.
    """
    job = db.get_sync_job(job_id)
    
    if job.status == 'completed':
        return {"message": "Job already completed"}
    
    # Resume from last successful point
    start_date = job.last_synced_date or job.start_date
    
    logger.info(
        f"Resuming sync from {start_date} "
        f"({job.chunks_completed}/{job.chunks_total} chunks done)"
    )
    
    # Continue with remaining chunks
    return await sync_large_range(
        metric_name=job.metric_name,
        start_date=start_date,
        end_date=job.end_date
    )
```

#### 4. Error Types & Handling

```python
class FulcraAPIError(Exception):
    """Base exception for Fulcra API errors."""
    pass

class RateLimitError(FulcraAPIError):
    """API rate limit exceeded."""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds to wait

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
```

#### 5. User-Facing Error Messages

**Good error messages help users understand what happened:**

```python
def format_sync_error(error: Exception, context: dict) -> str:
    """
    Convert technical errors into user-friendly messages.
    """
    if isinstance(error, RateLimitError):
        return (
            f"⚠️ Fulcra API rate limit reached. "
            f"Pausing for {error.retry_after}s then resuming automatically. "
            f"Progress saved: {context['completed']}/{context['total']} chunks."
        )
    
    elif isinstance(error, AuthenticationError):
        return (
            f"🔐 Authentication expired. "
            f"Please re-authenticate with Fulcra. "
            f"Progress saved: {context['completed']}/{context['total']} chunks."
        )
    
    elif isinstance(error, APITimeoutError):
        return (
            f"⏱️ Request timed out. "
            f"This can happen with large date ranges. "
            f"Retrying with smaller chunks... "
            f"Progress: {context['completed']}/{context['total']} chunks."
        )
    
    elif isinstance(error, DataNotFoundError):
        return (
            f"📭 No data available for the requested time range. "
            f"This might mean you didn't have a device recording data then."
        )
    
    else:
        return (
            f"❌ Unexpected error: {str(error)} "
            f"Progress saved: {context['completed']}/{context['total']} chunks. "
            f"You can resume this sync later with: health_db_resume_sync('{context['job_id']}')"
        )
```

#### 6. Configuration

**Environment variables for tuning:**

```bash
# Error handling
FULCRA_CHUNK_SIZE_DAYS=7                # Chunk size for large requests
FULCRA_MAX_RETRIES=3                    # Retry attempts per chunk
FULCRA_RETRY_BASE_DELAY=1.0             # Starting delay for backoff
FULCRA_REQUEST_TIMEOUT=30               # Timeout per request (seconds)
FULCRA_RATE_LIMIT_DELAY=0.5             # Delay between chunks (seconds)

# Safety limits
FULCRA_MAX_CHUNK_SIZE_DAYS=30           # Don't allow chunks larger than 30 days
FULCRA_MAX_CONCURRENT_CHUNKS=3          # Max parallel chunk fetches
FULCRA_MAX_SYNC_DURATION_HOURS=2        # Kill sync after 2 hours
```

#### 7. Monitoring & Logging

**Track errors for debugging:**

```sql
CREATE TABLE error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    context TEXT,                       -- JSON with request details
    resolved BOOLEAN DEFAULT 0,
    resolved_at REAL
);
```

**Log all errors:**

```python
def log_error(
    error_type: str,
    error_message: str,
    context: dict = None
):
    """Log error to database for later analysis."""
    db.execute(
        """
        INSERT INTO error_log (timestamp, error_type, error_message, context)
        VALUES (?, ?, ?, ?)
        """,
        (time.time(), error_type, error_message, json.dumps(context))
    )
    
    # Also log to stderr
    logger.error(
        f"{error_type}: {error_message}",
        extra={"context": context}
    )
```

#### 8. Testing Error Scenarios

**Test cases to implement:**

- [ ] Rate limit hit mid-sync (mock rate limit after N requests)
- [ ] Auth token expires during sync (mock token expiry)
- [ ] Network timeout on large request (mock timeout)
- [ ] Partial chunk success (some chunks fail, others succeed)
- [ ] Resume interrupted sync from last checkpoint
- [ ] Exponential backoff behavior
- [ ] Error message clarity for users

## Implementation Stages (Revised)

### Stage 1: Create Health Database Module ⬜
**File**: `fulcra_mcp/health_db.py`

**Features**:
- SQLite database with proper health data schema
- Insert methods for metrics, workouts, sleep, location
- Query methods with time range support
- Gap detection for smart fetching
- Export capabilities (CSV, JSON)

**Commit**: `git commit -m "Add health database module with permanent storage"`

**Checklist**:
- [ ] Create health_db.py module
- [ ] Implement database schema
- [ ] Add insert/upsert methods per data type
- [ ] Add query methods with time range filtering
- [ ] Implement gap detection logic
- [ ] Add data export functions
- [ ] Add database statistics/info methods

---

### Stage 2: Smart Fetch Wrapper ⬜
**File**: `fulcra_mcp/smart_fetch.py`

**Features**:
- Check local database before calling API
- Only fetch missing data (gaps)
- Handle "recent data" exception (last 2 hours)
- Log all fetch operations
- Graceful fallback to direct API if db unavailable

**Commit**: `git commit -m "Add smart fetch wrapper with gap detection"`

**Checklist**:
- [ ] Create smart_fetch.py
- [ ] Implement local-first query logic
- [ ] Add gap detection integration
- [ ] Handle recent data re-fetching
- [ ] Add fetch logging
- [ ] Test gap detection with mock data

---

### Stage 3: Configuration ⬜

**Environment Variables**:
```bash
FULCRA_DB_ENABLED="true"                    # Enable local database
FULCRA_DB_PATH="~/.fulcra_health_db"        # Database location
FULCRA_DB_RECENT_DATA_HOURS="2"             # How recent = "might change"
FULCRA_DB_AUTO_EXPORT="false"               # Auto-export on query
FULCRA_DB_EXPORT_PATH="~/fulcra_exports"    # Export location
```

**Commit**: `git commit -m "Add configuration for health database"`

---

### Stage 4: Integrate with Tools ⬜

**Modified behavior for each tool**:

#### `get_metric_time_series`
```python
@smart_fetch(db_table="health_metrics")
async def get_metric_time_series(metric_name, start_time, end_time, ...):
    # Wrapper handles:
    # 1. Check local DB
    # 2. Identify gaps
    # 3. Fetch only gaps from API
    # 4. Store in DB
    # 5. Return combined results
```

#### `get_workouts`
```python
@smart_fetch(db_table="workouts")
async def get_workouts(start_time, end_time):
    # Same pattern
```

#### `get_sleep_cycles`
```python
@smart_fetch(db_table="sleep_cycles")
async def get_sleep_cycles(start_time, end_time, ...):
    # Same pattern
```

#### `get_user_info` (metadata - different handling)
```python
async def get_user_info():
    # Check metadata_cache with 24-hour expiry
    cached = db.get_metadata("user_info", max_age=86400)
    if cached:
        return cached
    
    # Fetch and store
    data = await fulcra.get_user_info()
    db.set_metadata("user_info", data)
    return data
```

**Commit**: `git commit -m "Integrate health database with all tools"`

---

### Stage 5: Add Database Management Tools ⬜

**New MCP Tools**:

```python
@mcp.tool()
async def health_db_stats() -> str:
    """Get statistics about local health database."""
    return {
        "database_size_mb": ...,
        "total_records": ...,
        "records_by_type": {...},
        "date_range": {"oldest": ..., "newest": ...},
        "last_sync": ...
    }

@mcp.tool()
async def health_db_export(
    format: str = "csv",  # csv, json, parquet
    output_path: str = None
) -> str:
    """Export health database to file."""
    # Export all data or specific tables

@mcp.tool()
async def health_db_sync_range(
    start_date: str,
    end_date: str,
    metrics: list[str] = None
) -> str:
    """Proactively sync a date range to local database."""
    # Useful for "download all my 2025 data"

@mcp.tool()
async def health_db_clear(
    confirm: bool = False,
    older_than_days: int = None
) -> str:
    """Clear local health database (requires confirmation)."""
    # Safety feature if database gets too large
```

**Commit**: `git commit -m "Add database management tools"`

---

### Stage 6: Query Optimization ⬜

**Features**:
- Batch inserts for large datasets
- Connection pooling
- Query result caching (in-memory, not SQLite)
- Compression for large JSON fields
- Indexing strategy

**Commit**: `git commit -m "Optimize database performance"`

---

## Usage Examples

### Initial Setup
```
User: "Download all my health data from 2025"

Claude:
- Calls health_db_sync_range("2025-01-01", "2025-12-31")
- Fetches all data types from Fulcra (one time, might take a while)
- Stores locally in SQLite
- Reports: "Downloaded 50,000 records across 12 months"
```

### Daily Use (Fast!)
```
User: "Show me my sleep patterns for the last 6 months"

Claude:
- Checks local database: Has data for 5.5 months
- Identifies gap: Missing last 2 weeks
- Fetches only 2 weeks from API
- Returns analysis instantly from local data
- User sees results in <1 second instead of 10+ seconds
```

### Repeated Queries (Instant!)
```
User: "Actually, show me that same sleep data but analyzed differently"

Claude:
- All data already local
- Zero API calls
- Instant response
```

### Export for Other Tools
```
User: "Export my heart rate data as CSV"

Claude:
- health_db_export("csv", "heart_rate_2025.csv")
- User can now use Excel, Python, R, etc.
```

## Database Directory Structure

```
~/.fulcra_health_db/
  ├── fulcra_health.db          # Main SQLite database
  ├── exports/                  # Auto-exports if enabled
  │   ├── heart_rate_2025.csv
  │   └── sleep_2025.json
  └── backups/                  # Optional automatic backups
      └── fulcra_health_20260118.db
```

## Benefits of This Approach

1. **Minimal API Calls**: Fetch each piece of data exactly once
2. **Fast Queries**: Local SQLite is 10-100x faster than API
3. **Offline Capable**: Work without internet after initial sync
4. **Reusable**: Database can be used by other tools (pandas, R, Excel)
5. **Privacy**: All your health data stays on your machine
6. **Future-Proof**: Even if Fulcra changes API, you have your data
7. **Analysis Ready**: Structured database perfect for data science

## Key Design Decisions

### 1. Recent Data Handling (Last 2 Hours)
**Problem**: Data from the last 2 hours might still be updating
**Solution**: 
- Queries including last 2 hours: Always re-fetch that portion
- Historical data: Never re-fetch unless explicitly requested

### 2. Duplicate Prevention
**Strategy**: `UNIQUE` constraints on timestamp/ID combinations
- Use `INSERT OR IGNORE` for bulk imports
- Use `INSERT OR REPLACE` for recent data updates

### 3. Storage Size Management
**Estimates**:
- 1 year of sleep data: ~365 records × 50KB = ~18MB
- 1 year of heart rate (1-min intervals): ~500K records × 100 bytes = ~50MB
- Total for comprehensive data: ~200-500MB per year

**Management**:
- Configurable retention policies
- Compression for large fields
- Export old data and archive

### 4. Data Integrity
- Transaction support for batch operations
- Write-ahead logging (WAL mode) for concurrency
- Regular integrity checks
- Optional automated backups

## Testing Strategy

### Test Scenarios

1. **Cold Start**: Empty database, fetch large range
2. **Gap Filling**: Partial data, fetch only gaps
3. **Recent Data**: Data from last hour (should re-fetch)
4. **Historical Query**: Data from 6 months ago (pure local)
5. **Export**: Verify exported data matches database
6. **Concurrent Access**: Multiple queries at once

### Performance Benchmarks

Compare:
- API query time vs. local query time
- Number of API calls (before vs. after)
- Database size growth rate

## Future Enhancements

- [ ] Automatic data validation against known ranges
- [ ] Data quality checks (detect outliers, missing intervals)
- [ ] Multi-device sync (sync SQLite across devices)
- [ ] Compression for old data
- [ ] Built-in visualization from local database
- [ ] Incremental backup system
- [ ] Data anonymization for sharing/research

---

## Migration Path

For users who already have the "cache" version:
```sql
-- Migrate from cache_entries to health_metrics
INSERT INTO health_metrics (...)
SELECT ... FROM cache_entries WHERE tool_name = 'get_metric_time_series';

-- Set expires_at to NULL (make permanent)
UPDATE health_metrics SET expires_at = NULL;
```

---

## Notes

This is a fundamental shift from "temporary cache" to "personal health data warehouse". The goal is to build a local copy of your Fulcra data that:
- Minimizes API calls (respects rate limits)
- Enables fast, offline analysis
- Creates a reusable data asset
- Preserves your health data long-term
