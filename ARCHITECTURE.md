# Architecture: Fulcra Context MCP

## Overview

Fulcra Context MCP is a **personal health data warehouse** built on top of the Fulcra API. Unlike a simple API wrapper, it provides intelligent local caching, gap detection, and permanent storage of your health data in SQLite.

### Key Features

- **Local SQLite Database**: Permanent storage, not temporary cache
- **Smart Gap Detection**: Minimal API calls by fetching only missing data
- **Offline Capable**: Query your health data without internet
- **Data Export**: Export to CSV, JSON for external analysis
- **MCP Protocol**: Works with Claude Desktop and other MCP clients

---

## Core Principles

1. **Immutable Data Philosophy**
   Health data doesn't change once recorded - fetch once, store forever

2. **Minimize API Calls**
   Only fetch data that doesn't exist locally using smart gap detection

3. **Recent Data Exception**
   Data from the last 2 hours may still be updating and is re-fetched

4. **Reusable Archive**
   SQLite database can be used by other tools (pandas, R, Excel, etc.)

---

## System Architecture

```
┌─────────────────┐
│  MCP Client     │  (Claude Desktop, CLI)
│  (User)         │
└────────┬────────┘
         │ MCP Protocol
         │
┌────────▼────────┐
│ main.py         │  MCP Server
│ - 8 MCP Tools   │  - OAuth authentication
│ - OAuth flow    │  - FastAPI callback server
└────────┬────────┘
         │
         │
┌────────▼────────────────────────────┐
│ smart_fetch.py                      │
│ Intelligent Caching Layer           │
│ - Gap detection algorithm           │
│ - Local-first queries               │
│ - Fallback to API when needed       │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼──────┐
│ DB   │  │ Fulcra  │
│ (cache) │  API    │
└──────┘  └─────────┘
SQLite     HTTP/OAuth
```

---

## Database Schema

The SQLite database acts as a personal health data warehouse with the following tables:

### 1. health_metrics

Stores time-series biometric data (heart rate, running power, etc.)

```sql
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
```

### 2. workouts

Stores complete workout sessions with all statistics

```sql
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
```

### 3. sleep_cycles

Stores sleep cycle data with stages and quality metrics

```sql
CREATE TABLE sleep_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT UNIQUE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    data_json TEXT NOT NULL,          -- Full sleep data
    fetched_at REAL NOT NULL
);

CREATE INDEX idx_sleep_time ON sleep_cycles(start_time, end_time);
```

### 4. locations

Stores GPS location history

```sql
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
```

### 5. metadata_cache

Caches rarely-changing metadata (user info, metrics catalog)

```sql
CREATE TABLE metadata_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    expires_at REAL                   -- Optional expiry
);
```

### 6. error_log

Tracks errors for debugging and monitoring

```sql
CREATE TABLE error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    context TEXT,                     -- JSON with request details
    resolved BOOLEAN DEFAULT 0
);
```

---

## Smart Fetch Algorithm

The core innovation is **gap detection**: only fetching data that doesn't exist locally.

### High-Level Flow

```
1. Query comes in for data range [start_time, end_time]
2. Check local database for existing data
3. Identify gaps (missing time ranges)
4. Fetch only the gaps from Fulcra API
5. Insert new data into database
6. Return combined result (local + fetched)
```

### Gap Detection Logic

**Scenario 1: Empty Database (Cold Start)**
```
Requested: [T1 ────────────────────────── T2]
Local:     []
Gaps:      [T1 ────────────────────────── T2]  ← Fetch entire range
```

**Scenario 2: Partial Coverage**
```
Requested: [T1 ────────────────────────── T2]
Local:        [T3 ──── T4]
Gaps:      [T1─T3]            [T4 ─────── T2]  ← Fetch only gaps
```

**Scenario 3: Full Coverage**
```
Requested: [T1 ────────────────────────── T2]
Local:     [T1 ────────────────────────── T2]
Gaps:      []                                   ← No API call needed!
```

### Implementation

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

    # Gap before first record?
    if local_data[0].timestamp > requested_start:
        gaps.append((requested_start, local_data[0].timestamp))

    # Gaps between records?
    for i in range(len(local_data) - 1):
        gap_size = local_data[i+1].timestamp - local_data[i].timestamp
        if gap_size > expected_interval:
            gaps.append((local_data[i].timestamp, local_data[i+1].timestamp))

    # Gap after last record?
    if local_data[-1].timestamp < requested_end:
        gaps.append((local_data[-1].timestamp, requested_end))

    return gaps
```

---

## Error Handling & Reliability

### Challenges

When syncing large date ranges (months of data), we face:
- **Rate limiting**: Fulcra API throttles requests
- **Auth expiry**: OAuth tokens can expire mid-sync
- **Network timeouts**: Large requests fail
- **Partial failures**: Some chunks succeed, others fail

### Solutions

#### 1. Chunked Requests

Break large requests into manageable chunks (e.g., 7 days each):

```python
# Instead of: fetch(metric, "2025-01-01", "2025-12-31")
# Do this:
for chunk in split_into_weeks(start_date, end_date):
    fetch_and_store(metric, chunk.start, chunk.end)
```

#### 2. Exponential Backoff

Retry failed requests with increasing delays:

```python
delays = [1s, 2s, 4s, 8s]  # Exponential backoff
for attempt, delay in enumerate(delays):
    try:
        return fetch(...)
    except RateLimitError:
        await sleep(delay)
```

#### 3. Progress Tracking

Store sync progress in database for resumable operations:

```python
# If sync is interrupted, can resume from last successful chunk
job = {
    "job_id": "sync_2025_heart_rate",
    "last_synced": "2025-06-15",
    "chunks_completed": 24,
    "chunks_total": 52
}
```

#### 4. Error Classification

Different error types get different handling:

| Error Type | Action |
|------------|--------|
| RateLimitError | Wait retry_after seconds, then retry |
| AuthenticationError | Re-authenticate, then retry |
| TimeoutError | Retry with exponential backoff |
| DataNotFoundError | Skip chunk, continue with next |
| Unknown | Log error, continue with next chunk |

---

## MCP Tools

The server exposes 8 MCP tools to clients:

### Data Retrieval

1. **get_user_info()** - User profile and metadata
2. **get_metric_time_series()** - Biometric time-series data
3. **get_metric_samples()** - Raw biometric samples
4. **get_workouts()** - Workout sessions
5. **get_sleep_cycles()** - Sleep data
6. **get_location_time_series()** - GPS location history

### Database Management

7. **health_db_stats()** - Database size, record counts, coverage
8. **health_db_export()** - Export to CSV/JSON
9. **health_db_sync_range()** - Proactive sync of date range
10. **health_db_clear()** - Clear database (with confirmation)

---

## Data Flow Example

**User Query**: "Show me my heart rate from last week"

```
1. MCP Client (Claude Desktop)
   └─> Calls: get_metric_time_series("HeartRate", last_week_start, last_week_end)

2. smart_fetch.py
   ├─> Check local database
   │   └─> Found: 60% of data (Mon-Thu)
   ├─> Identify gaps: Fri-Sun
   ├─> Fetch gaps from Fulcra API
   │   └─> fetch_with_retry("HeartRate", friday, sunday)
   ├─> Insert new data to database
   └─> Return: Combined result (local Mon-Thu + fetched Fri-Sun)

3. health_db.py
   ├─> insert_metrics(friday_sunday_data)
   └─> query_metrics(last_week_start, last_week_end)
       └─> Returns: Full week of data

4. User receives: 7 days of heart rate data
   - 60% from local cache (instant)
   - 40% fetched from API (few seconds)
```

**Next Query** (same data):
```
1. Check local database → 100% coverage
2. No API call needed
3. Return cached data instantly
```

---

## Performance Characteristics

### First Query (Cold Start)

| Metric | Performance |
|--------|-------------|
| API calls | 1-N (depending on chunking) |
| Response time | 5-30 seconds (network dependent) |
| Database writes | All fetched records |
| Result | Full dataset returned |

### Subsequent Queries (Warm)

| Metric | Performance |
|--------|-------------|
| API calls | 0 (if fully cached) |
| Response time | <100ms (SQLite query) |
| Database writes | 0 |
| Result | Full dataset from cache |

### Partial Coverage

| Metric | Performance |
|--------|-------------|
| API calls | Only for gaps |
| Response time | Proportional to gap size |
| Database writes | Only new records |
| Result | Combined (cached + fetched) |

---

## Storage Estimates

| Data Type | 1 Year | Notes |
|-----------|--------|-------|
| Heart rate (1-min) | ~50 MB | 525,600 records |
| Sleep cycles | ~18 MB | 365 nights × ~50 KB |
| Workouts | ~10 MB | Variable by activity |
| Location (15-min) | ~35 MB | 35,040 points |
| **Total** | **~150-200 MB/year** | Highly compressible |

Database grows linearly with time. A 5-year archive would be ~1 GB.

---

## Key Design Decisions

### Why SQLite?

- **Portable**: Single file, works anywhere
- **Fast**: Local queries are 10-100x faster than API
- **Reliable**: ACID transactions, proven stability
- **Standard**: Works with pandas, R, Excel, etc.

### Why Permanent Storage?

- **Health data is immutable**: Once recorded, doesn't change
- **Minimize API costs**: Fulcra may have rate limits/pricing
- **Offline capability**: Work without internet
- **Data ownership**: Your data stays local

### Why Gap Detection?

- **Efficiency**: Only fetch what's missing
- **Reliability**: Partial failures don't lose progress
- **Flexibility**: Supports incremental updates
- **User experience**: Instant for cached data

---

## Future Enhancements

Potential improvements for future versions:

- **Data validation**: Detect outliers, missing intervals
- **Compression**: Reduce database size for old data
- **Multi-device sync**: Sync SQLite across devices
- **Built-in visualization**: Charts from local data
- **Anonymization**: Safe sharing for research
- **Incremental backups**: Automatic database backups

---

## References

- [Fulcra API Documentation](https://fulcra-dynamics.github.io/Context-API/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) file
