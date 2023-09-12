from __future__ import annotations

import datetime
from contextlib import closing
from posixpath import join as pjoin

import pandas as pd
import pytest
import pytz

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import config
from ibis.tests.util import assert_equal

pytest.importorskip("impala")
thrift = pytest.importorskip("thrift")


def test_hdfs_connect_function_is_public():
    assert hasattr(ibis.impala, "hdfs_connect")


def test_raise_ibis_error_no_hdfs(env):
    con = ibis.impala.connect(
        host=env.impala_host, port=env.impala_port, auth_mechanism=env.auth_mechanism
    )

    # GH299
    with pytest.raises(com.IbisError):
        con.hdfs  # noqa: B018


def test_run_sql(con, test_data_db):
    table = con.sql(f"SELECT li.* FROM {test_data_db}.lineitem li")

    li = con.table("lineitem")
    assert isinstance(table, ir.Table)
    assert_equal(table.schema(), li.schema())

    expr = table.limit(10)
    result = expr.execute()
    assert len(result) == 10


def test_sql_with_limit(con):
    table = con.sql("SELECT * FROM functional_alltypes LIMIT 10")
    ex_schema = con.get_schema("functional_alltypes")
    assert_equal(table.schema(), ex_schema)


def test_raw_sql(con):
    query = "SELECT * from functional_alltypes limit 10"
    with closing(con.raw_sql(query)) as cur:
        rows = cur.fetchall()
    assert len(rows) == 10


def test_explain(con):
    t = con.table("functional_alltypes")
    expr = t.group_by("string_col").size()
    result = con.explain(expr)
    assert isinstance(result, str)


def test_get_schema(con, test_data_db):
    t = con.table("lineitem")
    schema = con.get_schema("lineitem", database=test_data_db)
    assert_equal(t.schema(), schema)


def test_result_as_dataframe(con, alltypes):
    expr = alltypes.limit(10)

    ex_names = list(expr.schema().names)
    result = con.execute(expr)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ex_names
    assert len(result) == 10


def test_adapt_scalar_array_results(con, alltypes):
    table = alltypes

    expr = table.double_col.sum()
    result = con.execute(expr)
    assert isinstance(result, float)

    with config.option_context("interactive", True):
        result2 = expr.execute()
        assert isinstance(result2, float)

    expr = (
        table.group_by("string_col").aggregate([table.count().name("count")]).string_col
    )

    result = con.execute(expr)
    assert isinstance(result, pd.Series)


def test_interactive_repr_call_failure(con):
    t = con.table("lineitem").limit(100000)

    t = t[t, t.l_receiptdate.cast("timestamp").name("date")]

    keys = [t.date.year().name("year"), "l_linestatus"]
    filt = t.l_linestatus.isin(["F"])
    expr = t[filt].group_by(keys).aggregate(t.l_extendedprice.mean().name("avg_px"))

    w2 = ibis.trailing_window(9, group_by=expr.l_linestatus, order_by=expr.year)

    metric = expr["avg_px"].mean().over(w2)
    enriched = expr[expr, metric]
    with config.option_context("interactive", True):
        repr(enriched)


def test_array_default_limit(con, alltypes):
    t = alltypes

    result = con.execute(t.float_col, limit=100)
    assert len(result) == 100


def test_limit_overrides_expr(con, alltypes):
    # #418
    t = alltypes
    result = con.execute(t.limit(10), limit=5)
    assert len(result) == 5


def test_limit_equals_none_no_limit(alltypes):
    t = alltypes

    with config.option_context("sql.default_limit", 10):
        result = t.execute(limit=None)
        assert len(result) > 10


def test_verbose_log_queries(con, test_data_db):
    queries = []

    with config.option_context("verbose", True):
        with config.option_context("verbose_log", queries.append):
            con.table("orders", database=test_data_db)

    # we can't make assertions about the length of queries, since the Python GC
    # could've collected a temporary pandas table any time between construction
    # of `queries` and the assertion
    expected = f"DESCRIBE `{test_data_db}`.`orders`"
    assert expected in queries


def test_sql_query_limits(con, test_data_db):
    table = con.table("nation", database=test_data_db)
    with config.option_context("sql.default_limit", 100000):
        # table has 25 rows
        assert len(table.execute()) == 25
        # comply with limit arg for Table
        assert len(table.execute(limit=10)) == 10
        # state hasn't changed
        assert len(table.execute()) == 25
        # non-Table ignores default_limit
        assert table.count().execute() == 25
        # non-Table doesn't observe limit arg
        assert table.count().execute(limit=10) == 25
    with config.option_context("sql.default_limit", 20):
        # Table observes default limit setting
        assert len(table.execute()) == 20
        # explicit limit= overrides default
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=23)) == 23
        # non-Table ignores default_limit
        assert table.count().execute() == 25
        # non-Table doesn't observe limit arg
        assert table.count().execute(limit=10) == 25
    # eliminating default_limit doesn't break anything
    with config.option_context("sql.default_limit", None):
        assert len(table.execute()) == 25
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=10000)) == 25
        assert table.count().execute() == 25
        assert table.count().execute(limit=10) == 25


def test_database_default_current_database(con):
    db = con.database()
    assert db.name == con.current_database


def test_close_drops_temp_tables(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")

    table = con.parquet_file(hdfs_path)

    name = table._qualified_name
    assert len(con.list_tables(like=name))
    con.close()

    assert not len(con.list_tables(like=name))


def test_set_compression_codec(con):
    old_opts = con.get_options()
    assert old_opts["COMPRESSION_CODEC"].upper() in ("NONE", "")

    con.set_compression_codec("snappy")
    opts = con.get_options()
    assert opts["COMPRESSION_CODEC"].upper() == "SNAPPY"

    con.set_compression_codec(None)
    opts = con.get_options()
    assert opts["COMPRESSION_CODEC"].upper() in ("NONE", "")


def test_disable_codegen(con):
    con.disable_codegen(False)
    opts = con.get_options()
    assert opts["DISABLE_CODEGEN"] == "0"

    con.disable_codegen()
    opts = con.get_options()
    assert opts["DISABLE_CODEGEN"] == "1"

    impala_con = con.con
    cur1 = impala_con.execute("SET")
    cur2 = impala_con.execute("SET")

    opts1 = dict(row[:2] for row in cur1.fetchall())
    cur1.release()

    opts2 = dict(row[:2] for row in cur2.fetchall())
    cur2.release()

    assert opts1["DISABLE_CODEGEN"] == "1"
    assert opts2["DISABLE_CODEGEN"] == "1"


def test_attr_name_conflict(con, tmp_db, temp_parquet_table, temp_parquet_table2):
    left = temp_parquet_table
    right = temp_parquet_table2

    assert left.join(right, ["id"]) is not None
    assert left.join(right, ["id", "name"]) is not None
    assert left.join(right, ["id", "files"]) is not None


@pytest.fixture
def con2(env):
    con = ibis.impala.connect(
        host=env.impala_host,
        port=env.impala_port,
        auth_mechanism=env.auth_mechanism,
    )
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()["DISABLE_CODEGEN"] == "1"
    return con


def test_day_of_week(con):
    date_var = ibis.literal(datetime.date(2017, 1, 1), type=dt.date)
    expr_index = date_var.day_of_week.index()
    result = con.execute(expr_index)
    assert result == 6

    expr_name = date_var.day_of_week.full_name()
    result = con.execute(expr_name)
    assert result == "Sunday"


def test_datetime_to_int_cast(con):
    timestamp = pytz.utc.localize(datetime.datetime(2021, 9, 12, 14, 45, 33, 0))
    d = ibis.literal(timestamp)
    result = con.execute(d.cast("int64"))
    assert result == pd.Timestamp(timestamp).value // 1000


def test_set_option_with_dot(con):
    con.set_options({"request_pool": "baz.quux"})
    result = con.get_options()
    assert result["REQUEST_POOL"] == "baz.quux"


def test_list_databases(con):
    assert con.list_databases()


def test_list_tables(con, test_data_db):
    assert con.list_tables(database=test_data_db)
    assert con.list_tables(like="*nat*", database=test_data_db)
