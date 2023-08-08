from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis
import ibis.common.exceptions as com
from ibis.backends.snowflake.tests.conftest import _get_url
from ibis.util import gen_name


@pytest.fixture
def temp_db(con):
    db = gen_name("tmp_db")
    con.raw_sql(f"CREATE DATABASE {db}")
    yield db
    con.raw_sql(f"DROP DATABASE {db}")


@pytest.fixture
def temp_schema(con, temp_db):
    schema = gen_name("tmp_schema")
    con.raw_sql(f"CREATE SCHEMA {temp_db}.{schema}")
    assert schema in con.list_schemas(database=temp_db)
    yield schema
    con.raw_sql(f"DROP SCHEMA {temp_db}.{schema}")


def test_cross_db_access(con, temp_db, temp_schema):
    table = gen_name("tmp_table")
    con.raw_sql(f'CREATE TABLE {temp_db}.{temp_schema}."{table}" ("x" INT)')
    t = con.table(table, schema=f"{temp_db}.{temp_schema}")
    assert t.schema() == ibis.schema(dict(x="int"))
    assert t.execute().empty


@pytest.fixture(scope="session")
def simple_con():
    return ibis.connect(_get_url())


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


def test_repeated_memtable_registration(simple_con, mocker):
    data = {"key": list("abc"), "value": [[1], [2], [3]]}
    expected = pd.DataFrame(data)
    t = ibis.memtable(data)

    spy = mocker.spy(simple_con, "_register_in_memory_table")

    n = 2

    for _ in range(n):
        tm.assert_frame_equal(simple_con.execute(t), expected)

    # assert that we called _register_in_memory_table exactly n times
    assert spy.call_count == n


def test_timestamp_tz_column(simple_con):
    t = simple_con.create_table(
        ibis.util.gen_name("snowflake_timestamp_tz_column"),
        schema=ibis.schema({"ts": "string"}),
        temp=True,
    ).mutate(ts=lambda t: t.ts.to_timestamp("YYYY-MM-DD HH24-MI-SS"))
    expr = t.ts
    assert expr.execute().empty


def test_create_schema(simple_con):
    schema = gen_name("test_create_schema")

    cur_schema = simple_con.current_schema
    cur_db = simple_con.current_database

    simple_con.create_schema(schema)

    assert simple_con.current_schema == cur_schema
    assert simple_con.current_database == cur_db

    simple_con.drop_schema(schema)

    assert simple_con.current_schema == cur_schema
    assert simple_con.current_database == cur_db


def test_create_database(simple_con):
    database = gen_name("test_create_database")
    cur_db = simple_con.current_database

    simple_con.create_database(database)
    assert simple_con.current_database == cur_db

    simple_con.drop_database(database)
    assert simple_con.current_database == cur_db


@pytest.fixture(scope="session")
def db_con():
    return ibis.connect(_get_url())


@pytest.fixture(scope="session")
def schema_con():
    return ibis.connect(_get_url())


def test_drop_current_db_not_allowed(db_con):
    database = gen_name("test_create_database")
    cur_db = db_con.current_database

    db_con.create_database(database)

    assert db_con.current_database == cur_db

    with db_con.begin() as c:
        c.exec_driver_sql(f'USE DATABASE "{database}"')

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        db_con.drop_database(database)

    with db_con.begin() as c:
        c.exec_driver_sql(f"USE DATABASE {cur_db}")

    db_con.drop_database(database)


def test_drop_current_schema_not_allowed(schema_con):
    schema = gen_name("test_create_schema")
    cur_schema = schema_con.current_schema

    schema_con.create_schema(schema)

    assert schema_con.current_schema == cur_schema

    with schema_con.begin() as c:
        c.exec_driver_sql(f'USE SCHEMA "{schema}"')

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        schema_con.drop_schema(schema)

    with schema_con.begin() as c:
        c.exec_driver_sql(f"USE SCHEMA {cur_schema}")

    schema_con.drop_schema(schema)
