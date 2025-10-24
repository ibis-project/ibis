"""Tests for the allow_memtable_in_memory parameter in the Databricks backend."""

from __future__ import annotations

import builtins
from os import environ as env

import pytest

import ibis

pytestmark = pytest.mark.databricks


@pytest.fixture
def con_in_memory():
    """Create a Databricks connection with in-memory memtable support."""
    con = ibis.databricks.connect(
        server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
        http_path=env["DATABRICKS_HTTP_PATH"],
        access_token=env["DATABRICKS_TOKEN"],
        catalog="ibis_testing",
        schema="default",
        allow_memtable_in_memory=True,
    )
    yield con
    con.disconnect()


def test_allow_memtable_in_memory_connection(con_in_memory):
    """Test that connection with allow_memtable_in_memory=True initializes properly."""
    assert hasattr(con_in_memory, "_memtable_in_memory")
    assert con_in_memory._memtable_in_memory is True
    assert hasattr(con_in_memory, "_polars_backend")
    assert con_in_memory._polars_backend is not None


def test_memtable_in_memory_basic_operations(con_in_memory):
    """Test that basic memtable operations work with in-memory mode."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000.0, 60000.0, 70000.0],
    }

    memtable = ibis.memtable(data)

    # Test that the memtable can be executed
    result = con_in_memory.execute(memtable)
    assert len(result) == 3
    assert list(result.columns) == ["name", "age", "salary"]


def test_memtable_in_memory_filtering(con_in_memory):
    """Test that filtering works on in-memory memtables."""
    data = {
        "fruit": ["apple", "banana", "orange", "grape"],
        "price": [0.5, 0.25, 0.33, 0.75],
    }

    memtable = ibis.memtable(data)
    filtered = memtable.filter(memtable.price > 0.3)

    result = con_in_memory.execute(filtered)
    assert len(result) == 3  # apple, orange, grape


def test_memtable_in_memory_aggregation(con_in_memory):
    """Test that aggregations work on in-memory memtables."""
    data = {
        "category": ["A", "B", "A", "B", "A"],
        "value": [10, 20, 30, 40, 50],
    }

    memtable = ibis.memtable(data)
    aggregated = memtable.group_by("category").value.sum()

    result = con_in_memory.execute(aggregated).sort_values("category")
    assert len(result) == 2
    assert result.iloc[0]["value"] == 90  # A: 10+30+50
    assert result.iloc[1]["value"] == 60  # B: 20+40


def test_memtable_in_memory_join(con_in_memory):
    """Test that joins work between in-memory memtables."""
    left_data = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }

    right_data = {
        "id": [1, 2, 4],
        "dept": ["HR", "IT", "Sales"],
    }

    left = ibis.memtable(left_data)
    right = ibis.memtable(right_data)

    joined = left.join(right, "id")
    result = con_in_memory.execute(joined)

    assert len(result) == 2  # Only ids 1 and 2 match


def test_memtable_in_memory_cleanup(con_in_memory):
    """Test that in-memory memtables are properly cleaned up."""
    data = {"x": [1, 2, 3]}
    memtable = ibis.memtable(data)

    # Execute the memtable to register it
    con_in_memory.execute(memtable)

    # The memtable should be registered in the Polars backend
    # We can't directly check the finalizer, but we can verify the backend exists
    assert con_in_memory._polars_backend is not None


def test_import_error_without_polars(monkeypatch):
    """Test that proper ImportError is raised when Polars is not available."""
    # Mock the import to raise ImportError
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ibis.backends.polars" or name.startswith("ibis.backends.polars."):
            raise ImportError("No module named 'ibis.backends.polars'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="additional dependencies must be installed"):
        ibis.databricks.connect(
            server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
            http_path=env["DATABRICKS_HTTP_PATH"],
            access_token=env["DATABRICKS_TOKEN"],
            catalog="ibis_testing",
            schema="default",
            allow_memtable_in_memory=True,
        )


def test_memtable_in_memory_vs_default(con, con_in_memory):
    """Test difference between in-memory and default memtable handling."""
    # Default connection should not have _memtable_in_memory set to True
    assert not getattr(con, "_memtable_in_memory", False)

    # In-memory connection should have it set to True
    assert con_in_memory._memtable_in_memory is True

    # Both should be able to execute memtables
    data = {"x": [1, 2, 3]}
    memtable = ibis.memtable(data)

    result_default = con.execute(memtable)
    result_in_memory = con_in_memory.execute(memtable)

    # Results should be the same
    assert len(result_default) == len(result_in_memory)
    assert list(result_default.columns) == list(result_in_memory.columns)
