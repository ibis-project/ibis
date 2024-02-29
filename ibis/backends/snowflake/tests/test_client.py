from __future__ import annotations

import json
import os

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis.backends.snowflake.tests.conftest import _get_url
from ibis.util import gen_name


@pytest.fixture
def temp_db(con):
    db = gen_name("tmp_db")

    con.create_database(db)
    assert db in con.list_databases()

    yield db

    con.drop_database(db)
    assert db not in con.list_databases()


@pytest.fixture
def temp_schema(con, temp_db):
    schema = gen_name("tmp_schema")

    con.create_schema(schema, database=temp_db)
    assert schema in con.list_schemas(database=temp_db)

    yield schema

    con.drop_schema(schema, database=temp_db)
    assert schema not in con.list_schemas(database=temp_db)


def test_cross_db_access(con, temp_db, temp_schema):
    table = gen_name("tmp_table")
    con.raw_sql(f'CREATE TABLE "{temp_db}"."{temp_schema}"."{table}" ("x" INT)').close()
    t = con.table(table, schema=temp_schema, database=temp_db)
    assert t.schema() == ibis.schema(dict(x="int"))
    assert t.execute().empty


def test_cross_db_create_table(con, temp_db, temp_schema):
    table_name = gen_name("tmp_table")
    data = pd.DataFrame({"key": list("abc"), "value": [[1], [2], [3]]})
    table = con.create_table(table_name, data, database=f"{temp_db}.{temp_schema}")
    queried_table = con.table(table_name, database=temp_db, schema=temp_schema)

    tm.assert_frame_equal(table.execute(), data)
    tm.assert_frame_equal(queried_table.execute(), data)


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

    db_con.raw_sql(f'USE DATABASE "{database}"').close()

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        db_con.drop_database(database)

    db_con.raw_sql(f'USE DATABASE "{cur_db}"').close()

    db_con.drop_database(database)


def test_drop_current_schema_not_allowed(schema_con):
    schema = gen_name("test_create_schema")
    cur_schema = schema_con.current_schema

    schema_con.create_schema(schema)

    assert schema_con.current_schema == cur_schema

    schema_con.raw_sql(f'USE SCHEMA "{schema}"').close()

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        schema_con.drop_schema(schema)

    schema_con.raw_sql(f'USE SCHEMA "{cur_schema}"').close()

    schema_con.drop_schema(schema)


def test_read_csv_options(con, tmp_path):
    path = tmp_path / "test_pipe.csv"
    path.write_text("a|b\n1|2\n3|4\n")

    t = con.read_csv(path, field_delimiter="|")

    assert t.schema() == ibis.schema(dict(a="int64", b="int64"))


def test_read_csv_https(con):
    t = con.read_csv(
        "https://storage.googleapis.com/ibis-tutorial-data/wowah_data/locations.csv",
        field_optionally_enclosed_by='"',
    )
    assert t.schema() == ibis.schema(
        {
            "Map_ID": "int64",
            "Location_Type": "string",
            "Location_Name": "string",
            "Game_Version": "string",
        }
    )
    assert t.count().execute() == 151


@pytest.fixture(scope="module")
def json_data():
    return [
        {"a": 1, "b": "abc", "c": [{"d": 1}]},
        {"a": 2, "b": "def", "c": [{"d": 2}]},
        {"a": 3, "b": "ghi", "c": [{"d": 3}]},
    ]


@pytest.mark.parametrize(
    "serialize",
    [
        param(lambda obj: "\n".join(map(json.dumps, obj)), id="ndjson"),
        param(json.dumps, id="json"),
    ],
)
def test_read_json(con, tmp_path, serialize, json_data):
    path = tmp_path / "test.json"
    path.write_text(serialize(json_data))

    t = con.read_json(path)

    assert t.schema() == ibis.schema(dict(a="int", b="string", c="array<json>"))
    assert t.count().execute() == len(json_data)


def test_read_parquet(con, data_dir):
    path = data_dir / "parquet" / "functional_alltypes.parquet"

    t = con.read_parquet(path)

    assert t.timestamp_col.type().is_timestamp()


def test_array_repr(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    t = con.tables.ARRAY_TYPES
    expr = t.x
    assert repr(expr)


def test_insert(con):
    name = gen_name("test_insert")

    t = con.create_table(
        name, schema=ibis.schema({"a": "int", "b": "string", "c": "int"}), temp=True
    )
    assert t.count().execute() == 0

    expected = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None], "c": [2, None, 3]})

    con.insert(name, ibis.memtable(expected))

    result = t.order_by("a").execute()

    tm.assert_frame_equal(result, expected)

    con.insert(name, expected)
    assert t.count().execute() == 6

    con.insert(name, expected, overwrite=True)
    assert t.count().execute() == 3


def test_compile_does_not_make_requests(con, mocker):
    astronauts = con.table("astronauts")
    expr = astronauts.year_of_selection.value_counts()
    spy = mocker.spy(con.con, "cursor")
    assert expr.compile() is not None
    assert spy.call_count == 0

    t = ibis.memtable({"a": [1, 2, 3]})
    assert con.compile(t) is not None
    assert spy.call_count == 0

    assert ibis.to_sql(t, dialect="snowflake") is not None
    assert spy.call_count == 0

    assert ibis.to_sql(expr) is not None
    assert spy.call_count == 0


# this won't be hit in CI, but folks can test locally
@pytest.mark.xfail(
    condition=os.environ.get("SNOWFLAKE_HOME") is None,
    reason="SNOWFLAKE_HOME is not set",
)
@pytest.mark.xfail(
    condition=os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME") is None,
    reason="SNOWFLAKE_DEFAULT_CONNECTION_NAME is not set",
)
def test_no_argument_connection():
    con = ibis.snowflake.connect()
    assert con.list_tables() is not None

    con = ibis.connect("snowflake://")
    assert con.list_tables() is not None


def test_struct_of_json(con):
    raw = {"a": [1, 2, 3], "b": "456"}
    lit = ibis.struct(raw)
    expr = lit.cast("struct<a: array<int>, b: json>")

    n = 5
    t = con.tables.functional_alltypes.mutate(lit=expr).limit(n).lit
    result = con.to_pyarrow(t)

    assert len(result) == n
    assert all(value == raw for value in result.to_pylist())
