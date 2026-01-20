#!/usr/bin/env python3
"""
Functional Test Runner for Fulcra Context MCP

Run integration tests against real Fulcra API to validate:
- JSON serialization of numpy/datetime types
- Database operations (insert, query, cache)
- Smart fetch gap detection
- Error handling and retry logic

Usage:
  python -m fulcra_mcp.test_runner              # Run all tests
  python -m fulcra_mcp.test_runner --test serialization  # Run specific category
  python -m fulcra_mcp.test_runner --verbose         # Detailed output
  python -m fulcra_mcp.test_runner --keep-db         # Don't clean up test DB
  python -m fulcra_mcp.test_runner --list            # List available tests
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Test categories and their corresponding pytest markers/paths
TEST_CATEGORIES = {
    "all": {
        "description": "Run all tests (default)",
        "pytest_args": ["fulcra_mcp/tests/"]
    },
    "serialization": {
        "description": "Numpy, datetime, JSON conversion tests",
        "pytest_args": ["fulcra_mcp/tests/test_serialization.py"]
    },
    "database": {
        "description": "CRUD operations for all data types",
        "pytest_args": [
            "fulcra_mcp/tests/test_integration.py::TestMetricTimeSeriesCRUD",
            "fulcra_mcp/tests/test_integration.py::TestWorkoutsStorage",
            "fulcra_mcp/tests/test_integration.py::TestSleepCyclesStorage",
            "fulcra_mcp/tests/test_integration.py::TestLocationData",
            "fulcra_mcp/tests/test_integration.py::TestMetadataCaching"
        ]
    },
    "gaps": {
        "description": "Smart fetch gap detection scenarios",
        "pytest_args": [
            "fulcra_mcp/tests/test_integration.py::TestSmartFetchGapDetection",
            "fulcra_mcp/tests/test_integration.py::TestSmartFetcherIntegration"
        ]
    },
    "export": {
        "description": "Export functionality tests",
        "pytest_args": [
            "fulcra_mcp/tests/test_integration.py::TestExportFunctionality"
        ]
    },
    "stats": {
        "description": "Database statistics and management",
        "pytest_args": [
            "fulcra_mcp/tests/test_integration.py::TestDatabaseStats",
            "fulcra_mcp/tests/test_integration.py::TestDatabaseClearing"
        ]
    }
}


def list_tests():
    """List all available test categories."""
    print("\nAvailable Test Categories:")
    print("=" * 70)
    for name, info in TEST_CATEGORIES.items():
        print(f"\n{name:15} - {info['description']}")
    print("\n" + "=" * 70)
    print("\nUsage: python -m fulcra_mcp.test_runner --test <category>")
    print()


def run_tests(category: str, verbose: bool = False, keep_db: bool = False, extra_args: list = None):
    """
    Run tests for the specified category.

    Args:
        category: Test category name (e.g., 'all', 'serialization', 'database')
        verbose: Enable verbose output
        keep_db: Don't clean up test database after tests
        extra_args: Additional arguments to pass to pytest
    """
    if category not in TEST_CATEGORIES:
        print(f"Error: Unknown test category '{category}'")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        sys.exit(1)

    test_info = TEST_CATEGORIES[category]

    print(f"\n{'=' * 70}")
    print(f"Running: {category} - {test_info['description']}")
    print(f"{'=' * 70}\n")

    # Build pytest command
    pytest_args = ["pytest"] + test_info["pytest_args"]

    # Add verbose flag
    if verbose:
        pytest_args.extend(["-v", "-s", "--log-cli-level=DEBUG"])
    else:
        pytest_args.append("-v")

    # Add additional args
    if extra_args:
        pytest_args.extend(extra_args)

    # Add markers to skip slow tests by default unless explicitly running integration tests
    if category != "all" and "--slow" not in pytest_args:
        pytest_args.extend(["-m", "not slow"])

    # Set environment variable to keep test database if requested
    import os
    if keep_db:
        os.environ["KEEP_TEST_DB"] = "1"
        print("Note: Test database will be preserved at ~/.fulcra_health_db/test_fulcra_health.db\n")

    # Run pytest
    print(f"Command: {' '.join(pytest_args)}\n")
    result = subprocess.run(pytest_args)

    # Print summary
    print(f"\n{'=' * 70}")
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. See output above for details.")
    print(f"{'=' * 70}\n")

    return result.returncode


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Functional Test Runner for Fulcra Context MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fulcra_mcp.test_runner                    # Run all tests
  python -m fulcra_mcp.test_runner --test serialization  # Run serialization tests
  python -m fulcra_mcp.test_runner --verbose           # Detailed output
  python -m fulcra_mcp.test_runner --keep-db           # Keep test database
  python -m fulcra_mcp.test_runner --list              # List test categories

Test Categories:
  all            - Run all tests (default)
  serialization  - Numpy, datetime, JSON conversion tests
  database       - CRUD operations for all data types
  gaps           - Smart fetch gap detection scenarios
  export         - Export functionality tests
  stats          - Database statistics and management
        """
    )

    parser.add_argument(
        "--test",
        "-t",
        default="all",
        help="Test category to run (default: all). Use --list to see available categories."
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed logging"
    )

    parser.add_argument(
        "--keep-db",
        "-k",
        action="store_true",
        help="Don't clean up test database after tests (useful for debugging)"
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available test categories and exit"
    )

    parser.add_argument(
        "--markers",
        "-m",
        help="Pytest marker expression (e.g., 'not slow', 'integration')"
    )

    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_tests()
        sys.exit(0)

    # Check for pytest installation
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pytest is not installed.")
        print("Install with: pip install -e .[test]")
        sys.exit(1)

    # Build extra args
    extra_args = []
    if args.markers:
        extra_args.extend(["-m", args.markers])
    if args.pytest_args:
        extra_args.extend(args.pytest_args)

    # Run tests
    exit_code = run_tests(
        category=args.test,
        verbose=args.verbose,
        keep_db=args.keep_db,
        extra_args=extra_args
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
