from __future__ import annotations

import pytest

import ibis


@pytest.fixture
def tmp_table(con):
    """Create a temporary table in the Databricks backend."""
    name = ibis.util.gen_name("databricks_test_json")
    yield name
    con.drop_table(name, force=True)


def test_create_table_from_json(con, tmp_table):
    json_col_schema = con.sql("SELECT TRY_PARSE_JSON('{}') AS message").schema()
    table = con.create_table(tmp_table, schema=json_col_schema)
    assert table.schema() == json_col_schema
    assert table.execute().empty
