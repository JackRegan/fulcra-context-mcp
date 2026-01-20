# Contributing to Fulcra Context MCP

Thank you for your interest in contributing! This document provides guidelines for contributing to the Fulcra Context MCP project.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Testing with Real Fulcra API](#testing-with-real-fulcra-api)
- [Reporting Issues](#reporting-issues)

---

## Development Setup

### Prerequisites

- **Python 3.12+**
- **uv** (recommended package manager)
- **Fulcra API credentials** (for integration testing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/paulregan/fulcra-context-mcp.git
   cd fulcra-context-mcp
   ```

2. **Install dependencies** using `uv`:
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Set up environment variables**:
   ```bash
   export FULCRA_CLIENT_ID="your_client_id"
   export FULCRA_CLIENT_SECRET="your_client_secret"
   export FULCRA_DB_ENABLED="true"
   export FULCRA_DB_PATH="~/.fulcra_health_db_dev"
   ```

   Alternatively, create a `.env` file (not tracked in git):
   ```
   FULCRA_CLIENT_ID=your_client_id
   FULCRA_CLIENT_SECRET=your_client_secret
   FULCRA_DB_ENABLED=true
   FULCRA_DB_PATH=~/.fulcra_health_db_dev
   ```

---

## Running Tests

### Quick Start

Run the test suite with pytest:

```bash
pytest fulcra_mcp/tests/
```

### Test Categories

#### Unit Tests (Fast, No API)

```bash
pytest fulcra_mcp/tests/ -m "not integration"
```

These tests use mocks and don't require Fulcra API access.

#### Integration Tests (Require Real API)

```bash
pytest fulcra_mcp/tests/ -m integration
```

**Important**: Integration tests require:
- Valid Fulcra API credentials
- Active OAuth authentication
- Real health data in your Fulcra account

#### Run Specific Tests

```bash
# Test a specific file
pytest fulcra_mcp/tests/test_serialization.py

# Test a specific function
pytest fulcra_mcp/tests/test_integration.py::test_workout_fetch

# Verbose output
pytest -v fulcra_mcp/tests/
```

### Test Runner Utility

Use the provided test runner for detailed output:

```bash
python fulcra_mcp/test_runner.py
```

This shows:
- Test progress with status
- Detailed failure information
- Summary statistics

---

## Code Style

### Type Hints

All functions should have type hints:

```python
def calculate_gap(start: datetime, end: datetime) -> timedelta:
    """Calculate time gap between two timestamps."""
    return end - start
```

### Docstrings

Use Google-style docstrings:

```python
def fetch_metric_data(metric_name: str, start_time: datetime, end_time: datetime) -> list[dict]:
    """
    Fetch metric data from Fulcra API.

    Args:
        metric_name: Name of the metric (e.g., "HeartRate")
        start_time: Start of the time range (inclusive)
        end_time: End of the time range (exclusive)

    Returns:
        List of metric records with timestamps and values

    Raises:
        ValueError: If metric_name is invalid
        AuthenticationError: If OAuth token is expired
    """
    # Implementation here
```

### Logging

Use structlog for consistent logging:

```python
import structlog

logger = structlog.get_logger()

logger.info("Fetching data", metric_name="HeartRate", start=start_time, end=end_time)
logger.warning("No data found", time_range=[start_time, end_time])
logger.error("API request failed", error=str(e), status_code=response.status_code)
```

### Error Handling

- Catch specific exceptions, not generic `Exception`
- Log errors with context
- Raise custom exceptions for domain errors

```python
try:
    data = await fulcra.fetch(metric, start, end)
except RateLimitError as e:
    logger.warning("Rate limited", retry_after=e.retry_after)
    await asyncio.sleep(e.retry_after)
    data = await fulcra.fetch(metric, start, end)
except AuthenticationError:
    logger.error("Authentication failed")
    raise
```

### Code Formatting

Use `black` for formatting (if adopted):

```bash
black fulcra_mcp/
```

---

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test improvements
- `refactor/` for code refactoring

### 2. Make Your Changes

- Write clear, focused commits
- Add tests for new functionality
- Update documentation as needed
- Ensure tests pass

### 3. Commit Messages

Use conventional commit format:

```
type(scope): short description

Longer description if needed.

- Bullet points for details
- Reference issues with #123
```

Examples:
```
feat(health_db): add data export to CSV format

- Implement CSV export in health_db.py
- Add export_to_csv() method
- Include column headers and proper formatting

Fixes #45
```

```
fix(smart_fetch): handle None timestamps gracefully

- Skip records with None timestamps during normalization
- Add logging for filtered records
- Prevent ValueError crashes

Closes #78
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Screenshots if UI changes
- Test results

---

## Testing with Real Fulcra API

### OAuth Flow

1. **Start the MCP server** in your development environment
2. **Authenticate** using the OAuth flow (opens browser)
3. **Run tests** that require authentication:
   ```bash
   pytest -m integration
   ```

### Test Data Requirements

Integration tests require:
- At least one workout in your Fulcra account
- At least one sleep cycle
- Some biometric data (heart rate, etc.)

If tests fail due to missing data, you can:
- Record some activities in your connected apps
- Wait for data to sync to Fulcra
- Skip integration tests: `pytest -m "not integration"`

### Clean Test Database

Before testing, clear the test database:

```python
from fulcra_mcp.health_db import HealthDatabase

db = HealthDatabase("~/.fulcra_health_db_test")
db.clear(confirm=True)
```

Or use the test fixture which creates a clean database automatically.

---

## Reporting Issues

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check the CHANGELOG** to see if it's already fixed
3. **Test with latest version** from main branch

### Issue Template

When creating an issue, include:

**Bug Reports**:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (Python version, OS, etc.)
- Relevant logs/error messages

**Feature Requests**:
- Description of the feature
- Use case / motivation
- Proposed implementation (if any)
- Alternative solutions considered

---

## Development Workflow

### Typical Flow

1. Pick an issue or feature to work on
2. Create a branch
3. Write failing tests (TDD approach)
4. Implement the feature/fix
5. Make tests pass
6. Update documentation
7. Commit and push
8. Create PR
9. Address review feedback
10. Merge!

### Local Testing Cycle

```bash
# Make changes
vim fulcra_mcp/smart_fetch.py

# Run tests
pytest fulcra_mcp/tests/ -v

# Check specific test
pytest fulcra_mcp/tests/test_integration.py::test_gap_detection -v

# Test with real API (if needed)
pytest -m integration

# Commit when green
git add fulcra_mcp/smart_fetch.py
git commit -m "fix(smart_fetch): improve gap detection"
```

---

## Questions?

- **GitHub Discussions**: Ask questions in [Discussions](https://github.com/paulregan/fulcra-context-mcp/discussions)
- **Issues**: Report bugs in [Issues](https://github.com/paulregan/fulcra-context-mcp/issues)
- **Email**: Contact the maintainer (check GitHub profile)

---

## License

By contributing, you agree that your contributions will be licensed under the same Apache 2.0 License that covers this project.

