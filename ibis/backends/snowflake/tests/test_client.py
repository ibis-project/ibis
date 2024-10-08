from __future__ import annotations

import json
import os
from collections import Counter

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
def temp_catalog(con):
    cat = gen_name("tmp_catalog")

    con.create_catalog(cat)
    assert cat in con.list_catalogs()

    yield cat

    con.drop_catalog(cat)
    assert cat not in con.list_catalogs()


@pytest.fixture
def temp_db(con, temp_catalog):
    database = gen_name("tmp_database")

    con.create_database(database, catalog=temp_catalog)
    assert database in con.list_databases(catalog=temp_catalog)

    yield database

    con.drop_database(database, catalog=temp_catalog)
    assert database not in con.list_databases(catalog=temp_catalog)


def test_cross_db_access(con, temp_catalog, temp_db):
    table = gen_name("tmp_table")
    con.raw_sql(
        f'CREATE TABLE "{temp_catalog}"."{temp_db}"."{table}" ("x" INT)'
    ).close()
    t = con.table(table, database=(temp_catalog, temp_db))
    assert t.schema() == ibis.schema(dict(x="int"))
    assert t.execute().empty


def test_cross_db_create_table(con, temp_catalog, temp_db):
    table_name = gen_name("tmp_table")
    data = pd.DataFrame({"key": list("abc"), "value": [[1], [2], [3]]})
    table = con.create_table(table_name, data, database=f"{temp_catalog}.{temp_db}")
    queried_table = con.table(table_name, database=(temp_catalog, temp_db))

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

    # assert that we called _register_in_memory_table exactly once
    spy.assert_called_once()


def test_timestamp_tz_column(simple_con):
    t = simple_con.create_table(
        ibis.util.gen_name("snowflake_timestamp_tz_column"),
        schema=ibis.schema({"ts": "string"}),
        temp=True,
    ).mutate(ts=lambda t: t.ts.as_timestamp("YYYY-MM-DD HH24-MI-SS"))
    expr = t.ts
    assert expr.execute().empty


def test_create_database(simple_con):
    database = gen_name("test_create_database")

    cur_database = simple_con.current_database
    cur_catalog = simple_con.current_catalog

    simple_con.create_database(database)

    assert simple_con.current_database == cur_database
    assert simple_con.current_catalog == cur_catalog

    simple_con.drop_database(database)

    assert simple_con.current_database == cur_database
    assert simple_con.current_catalog == cur_catalog


def test_create_catalog(simple_con):
    catalog = gen_name("test_create_catalog")
    cur_catalog = simple_con.current_catalog

    simple_con.create_catalog(catalog)
    assert simple_con.current_catalog == cur_catalog

    simple_con.drop_catalog(catalog)
    assert simple_con.current_catalog == cur_catalog


@pytest.fixture(scope="session")
def cat_con():
    return ibis.connect(_get_url())


@pytest.fixture(scope="session")
def db_con():
    return ibis.connect(_get_url())


def test_drop_current_catalog_not_allowed(cat_con):
    catalog = gen_name("test_create_catalog")
    cur_cat = cat_con.current_catalog

    cat_con.create_catalog(catalog)

    assert cat_con.current_catalog == cur_cat

    cat_con.raw_sql(f'USE DATABASE "{catalog}"').close()

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        cat_con.drop_catalog(catalog)

    cat_con.raw_sql(f'USE DATABASE "{cur_cat}"').close()

    cat_con.drop_catalog(catalog)


def test_drop_current_db_not_allowed(db_con):
    database = gen_name("test_create_database")
    cur_database = db_con.current_database

    db_con.create_database(database)

    assert db_con.current_database == cur_database

    db_con.raw_sql(f'USE SCHEMA "{database}"').close()

    with pytest.raises(com.UnsupportedOperationError, match="behavior is undefined"):
        db_con.drop_database(database)

    db_con.raw_sql(f'USE SCHEMA "{cur_database}"').close()

    db_con.drop_database(database)


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

    name = gen_name("test_insert")

    t = con.create_table(
        name,
        schema=ibis.schema(
            {
                "ID": "int",
                "NAME": "string",
                "SCORE_1": "float",
                "SCORE_2": "float",
                "SCORE_3": "float",
                "AGE": "int",
            }
        ),
        temp=True,
        overwrite=True,
    )
    con.insert(
        name,
        obj=[
            {
                "ID": 10000,
                "NAME": "....",
                "SCORE_1": 0.2,
                "SCORE_2": 0.5,
                "SCORE_3": 0.75,
                "AGE": 1000,
            }
        ],
    )
    assert t.columns == ("ID", "NAME", "SCORE_1", "SCORE_2", "SCORE_3", "AGE")
    assert t.count().execute() == 1


def test_compile_does_not_make_requests(con, mocker):
    astronauts = con.table("astronauts")
    expr = astronauts.year_of_selection.value_counts()
    spy = mocker.spy(con.con, "cursor")
    assert expr.compile() is not None
    spy.assert_not_called()

    t = ibis.memtable({"a": [1, 2, 3]})
    assert con.compile(t) is not None
    spy.assert_not_called()

    assert ibis.to_sql(t, dialect="snowflake") is not None
    spy.assert_not_called()

    assert ibis.to_sql(expr) is not None
    spy.assert_not_called()


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


@pytest.mark.parametrize(
    "database",
    ["IBIS_TESTING.INFORMATION_SCHEMA", ("IBIS_TESTING", "INFORMATION_SCHEMA")],
    ids=["dotted-path", "tuple"],
)
def test_list_tables_with_database(con, database):
    like_table = {
        "EVENT_TABLES",
        "EXTERNAL_TABLES",
        "HYBRID_TABLES",
        "TABLES",
        "TABLE_CONSTRAINTS",
        "TABLE_PRIVILEGES",
        "TABLE_STORAGE_METRICS",
    }
    tables = con.list_tables(database=database, like="TABLE")
    assert like_table.issubset(tables)


def test_timestamp_memtable(con):
    df = pd.DataFrame(
        {
            "ts": [
                pd.Timestamp("1970-01-01 00:00:00"),
                pd.Timestamp("1970-01-01 00:00:01"),
                pd.Timestamp("1970-01-01 00:00:02"),
            ]
        }
    )
    t = ibis.memtable(df)
    result = con.to_pandas(t)
    tm.assert_frame_equal(result, df)


def test_connect_without_snowflake_url():
    # We're testing here that the non-URL connection works.
    # Specifically that a `database` location passed in as "catalog/database"
    # will be parsed correctly
    user = os.getenv("SNOWFLAKE_USER")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    catalog = os.getenv("SNOWFLAKE_DATABASE")
    database = os.getenv("SNOWFLAKE_SCHEMA")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    if (password := os.getenv("SNOWFLAKE_PASSWORD")) is None:
        pytest.skip(reason="No snowflake password set, nothing to do here")

    database = "/".join(filter(None, (catalog, database)))

    nonurlcon = ibis.snowflake.connect(
        user=user,
        account=account,
        database=database,
        warehouse=warehouse,
        password=password,
    )

    assert nonurlcon.list_tables()


def test_table_unnest_with_empty_strings(con):
    t = ibis.memtable({"x": [["", ""], [""], [], None]})
    expected = Counter(["", "", "", None, None])
    expr = t.unnest(t.x)["x"]
    result = con.execute(expr)
    assert Counter(result.values) == expected


def test_insert_dict_variants(con):
    name = gen_name("test_insert_dict_variants")

    t = con.create_table(name, schema=ibis.schema({"a": "int", "b": "str"}), temp=True)
    assert len(t.execute()) == 0

    data = [{"a": 1, "b": "a"}, {"a": 2, "b": "b"}]

    con.insert(name, data)
    assert len(t.execute()) == 2

    con.insert(name, ibis.memtable(data))
    assert len(t.execute()) == 4
