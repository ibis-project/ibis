from __future__ import annotations

import os

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis
from ibis.util import guid


def test_cross_db_access(con):
    db, schema = f"tmp_db_{guid()}", f"tmp_schema_{guid()}"
    schema = f"{db}.{schema}"
    table = f"tmp_table_{guid()}"
    con.raw_sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    try:
        con.raw_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        try:
            con.raw_sql(f'CREATE TEMP TABLE {schema}."{table}" ("x" INT)')
            try:
                t = con.table(table, schema=schema)
                assert t.schema() == ibis.schema(dict(x="int"))
                assert t.execute().empty
            finally:
                con.raw_sql(f'DROP TABLE {schema}."{table}"')
        finally:
            con.raw_sql(f"DROP SCHEMA {schema}")
    finally:
        con.raw_sql(f"DROP DATABASE {db}")


@pytest.fixture(scope="session")
def simple_con():
    if (url := os.environ.get("SNOWFLAKE_URL")) is None:
        pytest.skip("no snowflake credentials")
    return ibis.connect(url)


@pytest.mark.parametrize(
    "data",
    [
        # raw
        {"key": list("abc"), "value": [[1], [2], [3]]},
        # dataframe
        pd.DataFrame({"key": list("abc"), "value": [[1], [2], [3]]}),
        # pyarrow table
        pa.Table.from_pydict({"key": list("abc"), "value": [[1], [2], [3]]}),
    ],
)
def test_basic_memtable_registration(simple_con, data):
    expected = pd.DataFrame({"key": list("abc"), "value": [[1], [2], [3]]})
    t = ibis.memtable(data)
    result = simple_con.execute(t)
    tm.assert_frame_equal(result, expected)
