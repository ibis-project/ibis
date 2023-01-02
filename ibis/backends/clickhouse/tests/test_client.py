import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import config
from ibis.backends.clickhouse.tests.conftest import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASS,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    IBIS_TEST_CLICKHOUSE_DB,
)

pytest.importorskip("clickhouse_driver")


def test_run_sql(con):
    query = 'SELECT * FROM ibis_testing.functional_alltypes'
    table = con.sql(query)

    fa = con.table('functional_alltypes')
    assert isinstance(table, ir.Table)
    assert table.schema() == fa.schema()

    expr = table.limit(10)
    result = expr.execute()
    assert len(result) == 10


def test_get_schema(con):
    t = con.table('functional_alltypes')
    schema = con.get_schema('functional_alltypes')
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


def test_limit_equals_none_no_limit(alltypes):
    with config.option_context('sql.default_limit', 10):
        result = alltypes.execute(limit=None)
        assert len(result) > 10


def test_verbose_log_queries(con):
    queries = []

    def logger(x):
        queries.append(x)

    with config.option_context('verbose', True):
        with config.option_context('verbose_log', logger):
            con.table('functional_alltypes')

    expected = 'DESCRIBE ibis_testing.functional_alltypes'

    assert len(queries) == 1
    assert queries[0] == expected


def test_sql_query_limits(alltypes):
    table = alltypes
    with config.option_context('sql.default_limit', 100000):
        # table has 25 rows
        assert len(table.execute()) == 7300
        # comply with limit arg for Table
        assert len(table.execute(limit=10)) == 10
        # state hasn't changed
        assert len(table.execute()) == 7300
        # non-Table ignores default_limit
        assert table.count().execute() == 7300
        # non-Table doesn't observe limit arg
        assert table.count().execute(limit=10) == 7300
    with config.option_context('sql.default_limit', 20):
        # Table observes default limit setting
        assert len(table.execute()) == 20
        # explicit limit= overrides default
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=23)) == 23
        # non-Table ignores default_limit
        assert table.count().execute() == 7300
        # non-Table doesn't observe limit arg
        assert table.count().execute(limit=10) == 7300
    # eliminating default_limit doesn't break anything
    with config.option_context('sql.default_limit', None):
        assert len(table.execute()) == 7300
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=10000)) == 7300
        assert table.count().execute() == 7300
        assert table.count().execute(limit=10) == 7300


def test_embedded_identifier_quoting(alltypes):
    t = alltypes

    expr = t[[(t.double_col * 2).name('double(fun)')]]['double(fun)'].sum()
    expr.execute()


@pytest.fixture(scope="session")
def tmpcon():
    return ibis.clickhouse.connect(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        password=CLICKHOUSE_PASS,
        database="tmptables",
        user=CLICKHOUSE_USER,
    )


@pytest.fixture
def temporary_alltypes(tmpcon):
    id = ibis.util.guid()
    table = f"temporary_alltypes_{id}"
    tmpcon.raw_sql(
        f"CREATE TABLE {table} AS {IBIS_TEST_CLICKHOUSE_DB}.functional_alltypes"
    )
    yield tmpcon.table(table)
    tmpcon.raw_sql(f"DROP TABLE {table}")


def test_insert(temporary_alltypes, df):
    temporary = temporary_alltypes
    records = df[:10]

    assert len(temporary.execute()) == 0
    temporary.insert(records)

    tm.assert_frame_equal(temporary.execute(), records)


def test_insert_with_less_columns(temporary_alltypes, df):
    temporary = temporary_alltypes
    records = df.loc[:10, ['string_col']].copy()
    records['date_col'] = None

    with pytest.raises(AssertionError):
        temporary.insert(records)


def test_insert_with_more_columns(temporary_alltypes, df):
    temporary = temporary_alltypes
    records = df[:10].copy()
    records['non_existing_column'] = 'raise on me'

    with pytest.raises(AssertionError):
        temporary.insert(records)


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
    assert tables == tables2


def test_list_tables_empty_database(con, worker_id):
    dbname = f"tmpdb_{worker_id}"
    con.raw_sql(f"CREATE DATABASE IF NOT EXISTS {dbname}")
    try:
        assert not con.list_tables(database=dbname)
    finally:
        con.raw_sql(f"DROP DATABASE IF EXISTS {dbname}")
