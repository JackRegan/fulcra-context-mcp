# Fulcra Context MCP: Personal Health Data Warehouse

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

> **Note:** This is a fork of [fulcradynamics/fulcra-context-mcp](https://github.com/fulcradynamics/fulcra-context-mcp) with significant enhancements for local data warehousing.
>
> Full credit to the Fulcra team for the original implementation. For the official version, visit [Fulcra's developer docs](https://fulcradynamics.github.io/developer-docs/mcp-server/).

## Overview

Transform your health data from an API dependency into a **personal data warehouse**. This MCP server provides intelligent local caching with SQLite, smart gap detection, and permanent storage of your Fulcra health data.

### Why This Fork?

**Official MCP**: Simple API wrapper, queries Fulcra on every request

**This Fork**: Personal health database with:
- ✅ **Local SQLite database** - Permanent storage, not temporary cache
- ✅ **Smart gap detection** - Only fetch missing data, minimize API calls
- ✅ **Offline access** - Query your health data without internet
- ✅ **10-100x faster** - Local queries vs. API calls
- ✅ **Data ownership** - Your data stays on your machine
- ✅ **Export tools** - CSV, JSON for external analysis
- ✅ **Persistent auth tokens** - Survive server restarts, 7-day expiry

### Key Features

- **8 MCP Tools**: Access metrics, workouts, sleep, location data
- **Intelligent Caching**: Fetch once, store forever (immutable data philosophy)
- **Gap Detection Algorithm**: Automatically identifies missing time ranges
- **Error Recovery**: Chunked requests, automatic retry, progress tracking
- **Database Management**: Stats, export, sync, clear operations
- **Privacy-Focused**: All data stored locally, no cloud dependencies

📖 **[Read the Architecture Guide](ARCHITECTURE.md)** to understand how it works

## Available Tools

### Data Retrieval Tools

- `get_user_info` - User profile and preferences
- `get_metrics_catalog` - List all available biometric metrics
- `get_metric_time_series` - Time-series data for any metric (with smart caching)
- `get_metric_samples` - Raw samples for detailed analysis
- `get_workouts` - Workout sessions with full statistics
- `get_sleep_cycles` - Sleep data with stages and quality metrics
- `get_location_at_time` - GPS location at specific timestamp
- `get_location_time_series` - Location history over time

### Database Management Tools (New!)

- `health_db_stats` - View database size, record counts, coverage
- `health_db_export` - Export data to CSV or JSON
- `health_db_sync_range` - Proactively sync large date ranges
- `health_db_clear` - Clear database (with confirmation)

## Setup

### Local Installation (This Fork)

This fork is designed for local development and includes planned enhancements like caching.

**Prerequisites:**
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- A Fulcra account with active subscription

**Installation:**

1. Clone this repository:
```bash
git clone https://github.com/paulregan/fulcra-context-mcp.git
cd fulcra-context-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Configure Claude Desktop by adding to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "fulcra_context": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/fulcra-context-mcp",
        "python",
        "-m",
        "fulcra_mcp.main"
      ],
      "env": {
        "FULCRA_ENVIRONMENT": "stdio"
      }
    }
  }
}
```

4. Restart Claude Desktop

5. On first use, you'll be prompted to authenticate with Fulcra

**Note:** Replace `/absolute/path/to/fulcra-context-mcp` with the actual path to your cloned repository.

### Using the Official Public MCP

If you prefer not to run locally, you can use Fulcra's official public MCP server. See the [official documentation](https://fulcradynamics.github.io/developer-docs/mcp-server/) for setup instructions and the latest configuration options.

## Environment Variables

- `FULCRA_ENVIRONMENT` - Set to `stdio` for local mode, otherwise runs as HTTP server
- `OIDC_CLIENT_ID` - OAuth client ID (for remote server mode)
- `FULCRA_OIDC_DOMAIN` - OAuth domain (for remote server mode)
- `FULCRA_API` - Fulcra API endpoint (for remote server mode)

## Development

### Running Locally

```bash
# Set environment for local stdio mode
export FULCRA_ENVIRONMENT=stdio

# Run the server
uv run python -m fulcra_mcp.main
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run --directory /path/to/fulcra-context-mcp python -m fulcra_mcp.main
```

## Testing

The project includes comprehensive functional tests to validate JSON serialization, database operations, and API integration.

### Quick Start

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
python -m fulcra_mcp.test_runner

# Run specific test category
python -m fulcra_mcp.test_runner --test serialization

# Run with verbose output
python -m fulcra_mcp.test_runner --verbose

# List available test categories
python -m fulcra_mcp.test_runner --list
```

### Test Categories

- **serialization** - Tests numpy array/scalar conversion, datetime handling, and JSON serialization
- **database** - Tests CRUD operations for metrics, workouts, sleep, and location data
- **gaps** - Tests smart fetch gap detection and data filling scenarios
- **export** - Tests CSV and JSON export functionality
- **stats** - Tests database statistics and management operations
- **all** - Runs the complete test suite (default)

### Using Pytest Directly

```bash
# Run all tests with pytest
pytest fulcra_mcp/tests/ -v

# Run specific test file
pytest fulcra_mcp/tests/test_serialization.py -v

# Run with detailed logging
pytest fulcra_mcp/tests/ -v -s --log-cli-level=DEBUG

# Run tests excluding slow integration tests
pytest fulcra_mcp/tests/ -v -m "not slow"
```

### Test Database

Tests use an isolated database at `~/.fulcra_health_db/test_fulcra_health.db` which is automatically cleaned up after tests. To preserve the test database for debugging:

```bash
python -m fulcra_mcp.test_runner --keep-db
```

### Requirements for Integration Tests

Some tests require real Fulcra API access:
- Valid Fulcra account with active subscription
- OAuth authentication completed
- Internet connection
- Environment variable `FULCRA_API` set to the API endpoint

Integration tests are marked with `@pytest.mark.integration` and can be skipped:

```bash
pytest fulcra_mcp/tests/ -v -m "not integration"
```

### What the Tests Validate

1. **JSON Serialization** - Ensures all data types (numpy arrays, datetime objects, nested JSON) serialize correctly
2. **Database Operations** - Verifies insert, query, upsert, and deletion operations work correctly
3. **Smart Fetch Gap Detection** - Tests that missing data ranges are identified and fetched efficiently
4. **Metadata Caching** - Validates TTL-based caching for user info and metrics catalog
5. **Export Functionality** - Ensures CSV and JSON exports produce valid output
6. **Error Handling** - Tests graceful handling of edge cases and invalid inputs

### Known Issues Being Tested

Recent bug fixes validated by these tests:
- ✅ Numpy arrays convert to Python lists
- ✅ Numpy scalars convert to Python floats/ints
- ✅ Datetime objects serialize to ISO8601 strings
- ✅ Fulcra field names (`start_date`) map correctly to database schema (`timestamp`)
- ✅ None values handled without errors
- ✅ `get_metric_samples` routed through smart fetcher for caching

### Test Coverage

Current test coverage includes:
- 12 test classes
- 50+ individual test cases
- All 8 MCP tools
- All 4 database data types (metrics, workouts, sleep, locations)
- Multiple gap detection scenarios
- Export and database management operations

## Quick Start

After installation (see [Setup](#setup) below):

1. **First-time authentication**:
   ```
   Ask Claude: "Access my Fulcra health data"
   # OAuth flow opens in browser
   ```

2. **Query your data**:
   ```
   "Show me last night's sleep data"
   "What workouts did I do this week?"
   "What was my heart rate during yesterday's run?"
   ```

3. **Manage your database**:
   ```
   "Show me my health database stats"
   "Export my 2025 data to CSV"
   "Download all my heart rate data from January"
   ```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, database schema, smart fetch algorithm
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and recent fixes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup, testing, code style
- **[LICENSE](LICENSE)** - Apache 2.0 license

## Troubleshooting

**MCP server not appearing:**
- Ensure Claude Desktop has been restarted after config changes
- Check that the path in the config is absolute and correct
- Verify `uv sync` completed successfully

**Authentication issues:**
- The first time you use the server, Fulcra will open a browser for OAuth
- After authenticating, tokens are persisted to `state/tokens.json` and survive server restarts (7-day expiry)
- Ensure you have an active Fulcra subscription
- Check that `FULCRA_ENVIRONMENT=stdio` is set in the config
- If tokens expire, delete `state/tokens.json` and re-authenticate

**Data not loading:**
- Verify your Fulcra account has data for the requested time period
- Check that metrics are enabled in your Fulcra preferences
- Review Claude Desktop logs for error messages

## Support & Contributing

### This Fork

- **Issues**: [GitHub Issues](https://github.com/paulregan/fulcra-context-mcp/issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

### Upstream (Official Fulcra MCP)

- **Repository**: [fulcradynamics/fulcra-context-mcp](https://github.com/fulcradynamics/fulcra-context-mcp)
- **Discord**: [Fulcra Community](https://discord.com/invite/aunahVEnPU)
- **Email**: support@fulcradynamics.com

## License

Apache-2.0 (same as upstream)

## Acknowledgments

This project is based on the excellent work by the Fulcra team. Visit [Fulcra Dynamics](https://fulcradynamics.com) to learn more about their personal health data platform.
