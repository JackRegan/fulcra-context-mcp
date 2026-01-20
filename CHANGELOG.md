# Changelog

All notable changes to Fulcra Context MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-01-19

### Fixed

- **Critical**: Filter out records with None/null timestamps to prevent ValueError crashes
  - Fulcra API sometimes returns records with missing timestamp fields
  - System now gracefully skips invalid records with logging
  - Prevents `ValueError: Cannot convert <class 'NoneType'> to timestamp`

- **JSON Serialization**: Handle numpy arrays and pandas scalars in JSON responses
  - Custom JSON encoder now handles numpy int64, float64, and arrays
  - Prevents `TypeError: Object of type int64 is not JSON serializable`

- **Field Name Mapping**: Normalize field names during record processing
  - Maps Fulcra API field names (`start_date`, `end_date`) to standard names (`start_time`, `end_time`)
  - Ensures consistent timestamp handling across all data types

- **Database Caching**: Workouts and sleep cycles now cache correctly
  - Fixed database insertion for workout and sleep data
  - Verified with 38 workouts successfully cached in testing

### Added

- **Error Logging**: Comprehensive error logging to database
  - All API errors, validation failures, and exceptions logged to `error_log` table
  - Includes error type, message, context, and timestamp for debugging

- **Debug Logging**: Enhanced logging for gap detection and fetching
  - Logs gaps identified, API responses, DataFrame conversions, and database operations
  - Helps diagnose data flow issues during development

- **Fallback Return Logic**: Return fetched data even if database re-query fails
  - If local database query returns empty, fall back to fetched records
  - Prevents data loss when insertion/query boundary issues occur

- **Health Database Management Tools**:
  - `health_db_stats()` - View database size, record counts, and coverage
  - `health_db_export()` - Export data to CSV or JSON
  - `health_db_sync_range()` - Proactively sync large date ranges
  - `health_db_clear()` - Clear database with confirmation safety

### Changed

- Improved error messages for debugging
- Better handling of empty API responses
- More detailed logging throughout data pipeline

### Known Issues

- Metric names are case-sensitive (must use PascalCase like "HeartRate" not "heart_rate")
- Use `get_metrics_catalog()` to find valid metric names
- Some metrics may return empty if ALL records have None timestamps (Fulcra API data quality issue)

## [0.1.2] - 2025-12-XX

### Added
- Initial smart caching implementation
- SQLite database for permanent storage
- Gap detection algorithm
- Basic MCP tools for data retrieval

### Fixed
- OAuth authentication flow
- Basic error handling

## [0.1.1] - 2025-11-XX

### Added
- Fork from upstream Fulcra Context MCP
- Basic local caching concept

## [0.1.0] - 2025-10-XX

### Added
- Initial implementation based on upstream
- OAuth integration
- Basic MCP server

---

## Upgrade Notes

### 0.1.3

If you experience crashes related to timestamps, this version fixes that issue. Clear your local database and re-sync to ensure clean data:

```python
health_db_clear(confirm=True)
health_db_sync_range("2025-01-01", "2025-12-31")
```

**Metric Names**: Ensure you use correct PascalCase metric names. Check available metrics:

```python
get_metrics_catalog()
```

---

## Links

- [GitHub Repository](https://github.com/paulregan/fulcra-context-mcp)
- [Upstream Repository](https://github.com/fulcra-dynamics/fulcra-context-mcp)
- [Issue Tracker](https://github.com/paulregan/fulcra-context-mcp/issues)

