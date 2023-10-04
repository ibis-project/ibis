from __future__ import annotations

import pytest
from datafusion import (
    SessionContext,
)

import ibis
from ibis.backends.conftest import TEST_TABLES


@pytest.fixture
def name_to_path(data_dir):
    return {
        table_name: data_dir / "parquet" / f"{table_name}.parquet"
        for table_name in TEST_TABLES
    }


def test_none_config():
    config = None
    conn = ibis.datafusion.connect(config)
    assert conn.list_tables() == []


def test_str_config(name_to_path):
    config = {name: str(path) for name, path in name_to_path.items()}
    conn = ibis.datafusion.connect(config)
    assert sorted(conn.list_tables()) == sorted(name_to_path)


def test_path_config(name_to_path):
    config = name_to_path
    conn = ibis.datafusion.connect(config)
    assert sorted(conn.list_tables()) == sorted(name_to_path)


def test_context_config(name_to_path):
    ctx = SessionContext()
    for name, path in name_to_path.items():
        ctx.register_parquet(name, str(path))
    conn = ibis.datafusion.connect(ctx)
    assert sorted(conn.list_tables()) == sorted(name_to_path)
