from __future__ import annotations

import os
from urllib.parse import quote_plus

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import udf
from ibis.backends.clickhouse.tests.conftest import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASS,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    IBIS_TEST_CLICKHOUSE_DB,
)
from ibis.backends.tests.errors import ClickHouseDatabaseError
from ibis.util import gen_name

cc = pytest.importorskip("clickhouse_connect")


def test_connection(con):
    assert con is not None
    assert isinstance(con, ibis.backends.clickhouse.Backend)


def test_run_sql(con):
    query = "SELECT * FROM ibis_testing.functional_alltypes"
    table = con.sql(query)

    fa = con.table("functional_alltypes")
    assert isinstance(table, ir.Table)
    assert table.schema() == fa.schema()

    expr = table.limit(10)
    result = expr.execute()
    assert len(result) == 10


def test_get_schema(con):
    t = con.table("functional_alltypes")
    schema = con.get_schema("functional_alltypes")
    assert t.schema() == schema


def test_result_as_dataframe(con, alltypes):
    expr = alltypes.limit(10)

    ex_names = list(expr.schema().names)
    result = con.execute(expr)

    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ex_names
    assert len(result) == 10


def test_array_default_limit(con, alltypes):
    result = con.execute(alltypes.float_col, limit=100)
    assert len(result) == 100


def test_limit_overrides_expr(con, alltypes):
    result = con.execute(alltypes.limit(10), limit=5)
    assert len(result) == 5


def test_limit_equals_none_no_limit(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options.sql, "default_limit", 10)
    result = alltypes.execute(limit=None)
    assert len(result) > 10


def test_verbose_log_queries(con, monkeypatch):
    queries = []

    monkeypatch.setattr(ibis.options, "verbose", True)
    monkeypatch.setattr(ibis.options, "verbose_log", queries.append)

    con.table("functional_alltypes")

    expected = "DESCRIBE functional_alltypes"

    # might be other queries in there, we only check that a describe table
    # query was logged
    assert expected in queries


def test_sql_query_limits_big(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options.sql, "default_limit", 100_000)

    # alltypes has 7300 rows
    assert len(alltypes.execute()) == 7300  # comply with limit arg for alltypes
    assert len(alltypes.execute(limit=10)) == 10
    # state hasn't changed
    assert len(alltypes.execute()) == 7300
    # non-alltypes ignores default_limit
    assert alltypes.count().execute() == 7300
    # non-alltypes doesn't observe limit arg
    assert alltypes.count().execute(limit=10) == 7300


def test_sql_query_limits_small(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options.sql, "default_limit", 20)

    # alltypes observes default limit setting
    assert len(alltypes.execute()) == 20
    # explicit limit= overrides default
    assert len(alltypes.execute(limit=15)) == 15
    assert len(alltypes.execute(limit=23)) == 23
    # non-alltypes ignores default_limit
    assert alltypes.count().execute() == 7300
    # non-alltypes doesn't observe limit arg
    assert alltypes.count().execute(limit=10) == 7300


def test_sql_query_limits_none(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options.sql, "default_limit", None)

    # eliminating default_limit doesn't break anything
    assert len(alltypes.execute()) == 7300
    assert len(alltypes.execute(limit=15)) == 15
    assert len(alltypes.execute(limit=10000)) == 7300
    assert alltypes.count().execute() == 7300
    assert alltypes.count().execute(limit=10) == 7300


def test_embedded_identifier_quoting(alltypes):
    t = alltypes

    expr = t.select((t.double_col * 2).name("double(fun)"))["double(fun)"].sum()
    expr.execute()


@pytest.fixture
def temporary_alltypes(con):
    table = gen_name("temporary_alltypes")
    con.raw_sql(f"CREATE TABLE {table} AS functional_alltypes")
    yield con.table(table)
    con.drop_table(table)


@pytest.mark.parametrize(
    "transform",
    [param(lambda df: df, id="pandas"), param(pa.Table.from_pandas, id="pyarrow")],
)
def test_insert(con, temporary_alltypes, df, transform):
    temporary = temporary_alltypes
    records = df[:10]

    assert temporary.count().execute() == 0

    con.insert(temporary.op().name, transform(records))

    tm.assert_frame_equal(temporary.execute(), records)


def test_insert_with_less_columns(con, temporary_alltypes, df):
    temporary = temporary_alltypes
    records = df.loc[:10, ["string_col"]].copy()
    records["date_col"] = None

    with pytest.raises(cc.driver.exceptions.DatabaseError):
        con.insert(temporary.op().name, records)


def test_insert_with_more_columns(con, temporary_alltypes, df):
    temporary = temporary_alltypes
    records = df[:10].copy()
    records["non_existing_column"] = "raise on me"

    with pytest.raises(cc.driver.exceptions.DatabaseError):
        con.insert(temporary.op().name, records)


@pytest.mark.parametrize(
    ("query", "expected_schema"),
    [
        (
            "SELECT 1 as a, 2 + dummy as b",
            ibis.schema(dict(a=dt.UInt8(nullable=False), b=dt.UInt16(nullable=False))),
        ),
        (
            "SELECT string_col, sum(double_col) as b FROM functional_alltypes GROUP BY string_col",
            ibis.schema(
                dict(
                    string_col=dt.String(nullable=True),
                    b=dt.Float64(nullable=True),
                )
            ),
        ),
    ],
)
def test_get_schema_using_query(con, query, expected_schema):
    result = con._get_schema_using_query(query)
    assert result == expected_schema


def test_list_tables_database(con):
    tables = con.list_tables()
    tables2 = con.list_tables(database=con.current_database)
    # some overlap, but not necessarily identical because
    # a database may have temporary tables added/removed between list_tables
    # calls
    assert set(tables) & set(tables2)


@pytest.fixture
def tmpcon(worker_id):
    dbname = f"clickhouse_database_{worker_id}"
    con = ibis.clickhouse.connect(
        host=os.environ.get("IBIS_TEST_CLICKHOUSE_HOST", "localhost"),
        user=os.environ.get("IBIS_TEST_CLICKHOUSE_USER", "ibis"),
        port=int(os.environ.get("IBIS_TEST_CLICKHOUSE_PORT", "8123")),
        password=os.environ.get("IBIS_TEST_CLICKHOUSE_PASSWORD", ""),
    )
    con.create_database(dbname, force=True)
    yield con
    con.drop_database(dbname, force=True)


def test_list_tables_empty_database(tmpcon):
    assert not tmpcon.list_tables()


@pytest.mark.parametrize("temp", [True, False], ids=["temp", "no_temp"])
def test_create_table_no_data(con, temp, temp_table):
    schema = ibis.schema(dict(a="!int", b="string"))
    t = con.create_table(temp_table, schema=schema, temp=temp, engine="Memory")
    assert t.execute().empty


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [None, "b", "c"]},
        pd.DataFrame({"a": [1, 2, 3], "b": [None, "b", "c"]}),
        pa.Table.from_pydict({"a": [1, 2, 3], "b": [None, "b", "c"]}),
    ],
    ids=["dict", "dataframe", "pyarrow table"],
)
@pytest.mark.parametrize(
    "engine",
    ["File(Native)", "File(Parquet)", "Memory"],
    ids=["native", "mem", "parquet"],
)
def test_create_table_data(con, data, engine, temp_table):
    schema = ibis.schema(dict(a="!int", b="string"))
    t = con.create_table(temp_table, obj=data, schema=schema, engine=engine)
    assert len(t.execute()) == 3


def test_create_table_with_properties(con, temp_table):
    data = pd.DataFrame({"a": list("abcde" * 20), "b": [1, 2, 3, 4, 5] * 20})
    n = len(data)
    t = con.create_table(
        temp_table,
        data,
        schema=ibis.schema(dict(a="string", b="!uint32")),
        order_by=["a", "b"],
        partition_by=["a"],
        sample_by=["b"],
        settings={"allow_nullable_key": "1"},
    )
    assert t.count().execute() == n


@pytest.mark.parametrize(
    "engine",
    [
        "File(Native)",
        param(
            "File(Parquet)",
            marks=pytest.mark.xfail(
                reason="Parquet file size is 0 bytes",
                raises=cc.driver.exceptions.DatabaseError,
            ),
        ),
        "Memory",
    ],
    ids=["native", "mem", "parquet"],
)
def test_truncate_table(con, engine, temp_table):
    t = con.create_table(temp_table, obj={"a": [1]}, engine=engine)
    assert len(t.execute()) == 1
    con.truncate_table(temp_table)
    assert t.execute().empty


@udf.scalar.builtin(name="arrayJaccardIndex")
def array_jaccard_index(a: dt.Array[dt.int64], b: dt.Array[dt.int64]) -> float: ...


@udf.scalar.builtin(name="arrayJaccardIndex")
def array_jaccard_index_no_input_types(a, b) -> float: ...


@udf.scalar.builtin
def arrayJaccardIndex(a: dt.Array[dt.int64], b: dt.Array[dt.int64]) -> float: ...


@pytest.mark.parametrize(
    "func",
    [
        array_jaccard_index,
        arrayJaccardIndex,
        array_jaccard_index_no_input_types,
    ],
)
def test_builtin_scalar_udf(con, func):
    expr = func([1, 2], [2, 3])
    result = con.execute(expr)
    expected = 1.0 / 3.0
    assert result == expected


@udf.agg.builtin
def entropy(a) -> float: ...


@udf.agg.builtin(name="sumKahan")
def sum_kahan(a: float) -> float: ...


@pytest.mark.parametrize("func", [entropy, sum_kahan])
def test_builtin_agg_udf(con, func):
    t = ibis.memtable(pd.DataFrame({"a": list(map(float, range(10)))}))

    expr = func(t.a)
    result = con.execute(expr)

    table_name = gen_name("agg_udf_test")
    expected = con.con.query_df(
        f"SELECT {expr.op().__func_name__}(a) FROM {table_name}",
        external_data=con._normalize_external_tables({table_name: t.op()}),
    ).squeeze()
    assert result == expected


def test_create_table_no_syntax_error(con):
    schema = ibis.schema(
        [
            ("play_id", "!string"),
            ("uid", "!int64"),
            ("feat1", "float64"),
            ("feat2", "float64"),
            ("feat3", "float64"),
            ("feat4", "float64"),
            ("feat5", "float64"),
            ("feat6", "float64"),
            ("feat7", "float64"),
            ("feat8", "float64"),
            ("feat9", "float64"),
            ("feat10", "float64"),
            ("feat11", "float64"),
            ("play_timestamp", "!int64"),
            ("tag_date", "!int32"),
        ]
    )
    t = con.create_table(gen_name("clickouse_temp_table"), schema=schema, temp=True)
    assert t.count().execute() == 0


def test_password_with_bracket():
    password = f"{os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')}[]"
    quoted_pass = quote_plus(password)
    host = os.environ.get("IBIS_TEST_CLICKHOUSE_HOST", "localhost")
    user = os.environ.get("IBIS_TEST_CLICKHOUSE_USER", "default")
    port = int(os.environ.get("IBIS_TEST_CLICKHOUSE_PORT", "8123"))
    with pytest.raises(
        cc.driver.exceptions.DatabaseError, match="password is incorrect"
    ):
        ibis.clickhouse.connect(host=host, user=user, port=port, password=quoted_pass)


def test_from_url(con):
    assert ibis.connect(
        f"clickhouse://{CLICKHOUSE_USER}:{CLICKHOUSE_PASS}@{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{IBIS_TEST_CLICKHOUSE_DB}"
    )


def test_from_url_with_kwargs(con):
    # since explicit kwargs take precedence, this passes, because we're passing
    # `database` explicitly, even though our connection string says to use a
    # random database
    database = ibis.util.gen_name("clickhouse_database")
    assert ibis.connect(
        f"clickhouse://{CLICKHOUSE_USER}:{CLICKHOUSE_PASS}@{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{database}",
        database=IBIS_TEST_CLICKHOUSE_DB,
    )


def test_invalid_port():
    port = 9999
    url = f"clickhouse://{CLICKHOUSE_USER}:{CLICKHOUSE_PASS}@{CLICKHOUSE_HOST}:{port}/{IBIS_TEST_CLICKHOUSE_DB}"
    with pytest.raises(cc.driver.exceptions.DatabaseError):
        ibis.connect(url)


def test_subquery_with_join(con):
    name = gen_name("clickhouse_tmp_table")

    s = con.create_table(name, pa.Table.from_pydict({"a": [1, 2, 3]}), temp=True)

    sql = f"""
    SELECT
      "o"."a"
    FROM (
      SELECT
        "w"."a"
      FROM "{name}" AS "s"
      INNER JOIN "{name}" AS "w"
      USING ("a")
    ) AS "o"
    """
    with pytest.raises(
        ClickHouseDatabaseError, match="Identifier 'o.a' cannot be resolved"
    ):
        # https://github.com/ClickHouse/ClickHouse/issues/66133
        con.sql(sql)

    # this works because we add the additional alias in the inner query
    w = s.view()
    expr = s.join(w, "a").select(a=w.a).select(b=lambda t: t.a + 1)
    result = expr.to_pandas()
    assert set(result["b"].tolist()) == {2, 3, 4}


def test_alias_column_ref(con):
    data = {"user_id": [1, 2, 3], "account_id": [4, 5, 6]}
    t = con.create_table(gen_name("clickhouse_temp_table"), data, temp=True)
    expr = t.alias("df").sql("select *, halfMD5(account_id) as id_md5 from df")

    result = expr.execute()

    assert len(result) == 3

    assert result.columns.tolist() == ["user_id", "account_id", "id_md5"]

    assert result.user_id.notnull().all()
    assert result.account_id.notnull().all()
    assert result.id_md5.notnull().all()


@pytest.mark.parametrize("method_name", ["to_pandas", "to_pyarrow"])
def test_query_cache(con, method_name):
    t = con.table("functional_alltypes")
    expr = t.count()

    method = getattr(expr, method_name)

    expected = method()
    result = method(settings={"use_query_cache": True})

    # test a bogus setting
    with pytest.raises(ClickHouseDatabaseError):
        method(settings={"ooze_query_cash": True})

    assert result == expected


def test_invalid_catalog_argument(con):
    with pytest.raises(
        exc.UnsupportedOperationError, match="`catalog` namespaces are not supported"
    ):
        con.get_schema("t", catalog="a", database="b")
