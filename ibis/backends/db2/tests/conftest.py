"""Pytest configuration and fixtures for DB2 backend tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session")
def db2_config():
    """
    Get DB2 connection configuration from environment variables.

    Returns
    -------
    dict
        Connection configuration
    """
    return {
        "database": os.getenv("DB2_DATABASE", "SAMPLE"),
        "hostname": os.getenv("DB2_HOSTNAME", "localhost"),
        "port": int(os.getenv("DB2_PORT", "50000")),
        "username": os.getenv("DB2_USERNAME", "db2inst1"),
        "password": os.getenv("DB2_PASSWORD", "password"),
        "schema": os.getenv("DB2_SCHEMA"),
    }


@pytest.fixture(scope="session")
def con(db2_config):
    """
    Create a DB2 backend connection for testing.

    Parameters
    ----------
    db2_config : dict
        Connection configuration

    Returns
    -------
    Backend
        Connected backend instance
    """
    from ibis.backends import db2

    try:
        backend = db2.connect(**db2_config)
        yield backend
        backend.disconnect()
    except Exception as e:
        pytest.skip(f"Could not connect to DB2: {e}")


@pytest.fixture
def test_table_name():
    """Generate a unique test table name."""
    import uuid

    return f"test_table_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "is_active": [True, True, False, True, False],
        }
    )


@pytest.fixture
def temp_table(con, test_table_name, sample_dataframe):
    """
    Create a temporary test table.

    Parameters
    ----------
    con : Backend
        Backend connection
    test_table_name : str
        Table name
    sample_dataframe : pd.DataFrame
        Sample data

    Yields
    ------
    str
        Table name
    """
    # Create table
    con.create_table(test_table_name, sample_dataframe)

    yield test_table_name

    # Cleanup
    try:
        con.drop_table(test_table_name, force=True)
    except Exception:
        pass


@pytest.fixture
def alltypes_table(con):
    """
    Create a table with all supported data types.

    Parameters
    ----------
    con : Backend
        Backend connection

    Yields
    ------
    str
        Table name
    """
    from datetime import date, datetime

    import pandas as pd

    table_name = "test_alltypes"

    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "bigint_col": [1000000, 2000000, 3000000],
            "float_col": [1.1, 2.2, 3.3],
            "double_col": [1.111, 2.222, 3.333],
            "string_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "timestamp_col": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 12, 0, 0),
                datetime(2024, 1, 3, 12, 0, 0),
            ],
        }
    )

    try:
        con.create_table(table_name, df, overwrite=True)
        yield table_name
    finally:
        try:
            con.drop_table(table_name, force=True)
        except Exception:
            pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires DB2)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark all tests in test_integration.py as integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
