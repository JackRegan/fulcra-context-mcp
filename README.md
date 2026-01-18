# fulcra-context-mcp: Local MCP Server for Fulcra Context Data

> **Note:** This is a fork of [fulcradynamics/fulcra-context-mcp](https://github.com/fulcradynamics/fulcra-context-mcp) customized for local development with enhanced caching capabilities.
> 
> Full credit to the Fulcra team for the original implementation. For the official version and documentation, visit [Fulcra's developer docs](https://fulcradynamics.github.io/developer-docs/mcp-server/).

## Overview

This MCP server provides tools to access your Fulcra Context health and activity data through the Fulcra API using [`fulcra-api`](https://github.com/fulcradynamics/fulcra-api-python).

**Key Features:**
- Access health metrics (sleep, workouts, heart rate, etc.)
- Query location and activity data
- Time-series data analysis
- Local caching for improved performance (planned)
- Privacy-focused local execution

## Available Tools

The MCP server provides the following tools:

- `get_user_info` - Get user profile and preferences
- `get_metrics_catalog` - List all available metrics
- `get_metric_time_series` - Get time-series data for a metric
- `get_metric_samples` - Get raw samples for a metric
- `get_workouts` - Retrieve workout data for a time period
- `get_sleep_cycles` - Get detailed sleep cycle information
- `get_location_at_time` - Get location at a specific time
- `get_location_time_series` - Get location data over time

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

## Planned Enhancements

This fork includes planned enhancements:

- **Local Caching**: SQLite-based caching of API responses for:
  - Faster query responses
  - Reduced API calls
  - Offline access to historical data
  - Better privacy (data stored locally)

- **Extended Metrics**: Additional data processing and aggregation

## Example Usage

Once configured in Claude Desktop, you can ask:

- "Show me last night's sleep data"
- "What workouts did I do this week?"
- "What was my heart rate during my run yesterday?"
- "Where was I at 3pm on Tuesday?"
- "Show me my step count trend for the past month"

## Troubleshooting

**MCP server not appearing:**
- Ensure Claude Desktop has been restarted after config changes
- Check that the path in the config is absolute and correct
- Verify `uv sync` completed successfully

**Authentication issues:**
- The first time you use the server, Fulcra will open a browser for OAuth
- Ensure you have an active Fulcra subscription
- Check that `FULCRA_ENVIRONMENT=stdio` is set in the config

**Data not loading:**
- Verify your Fulcra account has data for the requested time period
- Check that metrics are enabled in your Fulcra preferences
- Review Claude Desktop logs for error messages

## Support

For issues specific to this fork:
- Open an issue on this repository

For general Fulcra MCP questions:
- [Official GitHub Repository](https://github.com/fulcradynamics/fulcra-context-mcp)
- [Fulcra Discord](https://discord.com/invite/aunahVEnPU)
- Email: support@fulcradynamics.com

## License

Apache-2.0 (same as upstream)

## Acknowledgments

This project is based on the excellent work by the Fulcra team. Visit [Fulcra Dynamics](https://fulcradynamics.com) to learn more about their personal health data platform.
