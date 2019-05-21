import pytest
import pandas as pd

import ibis
import ibis.config as config
import ibis.expr.types as ir
import pandas.util.testing as tm

from ibis import literal as L
from ibis.compat import StringIO

pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


def test_get_table_ref(db):
    table = db.functional_alltypes
    assert isinstance(table, ir.TableExpr)

    table = db['functional_alltypes']
    assert isinstance(table, ir.TableExpr)


def test_run_sql(con, db):
    query = 'SELECT * FROM {0}.`functional_alltypes`'.format(db.name)
    table = con.sql(query)

    fa = con.table('functional_alltypes')
    assert isinstance(table, ir.TableExpr)
    assert table.schema() == fa.schema()

    expr = table.limit(10)
    result = expr.execute()
    assert len(result) == 10


def test_get_schema(con, db):
    t = con.table('functional_alltypes')
    schema = con.get_schema('functional_alltypes', database=db.name)
    assert t.schema() == schema


def test_result_as_dataframe(con, alltypes):
    expr = alltypes.limit(10)

    ex_names = expr.schema().names
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


def test_verbose_log_queries(con, db):
    queries = []

    def logger(x):
        queries.append(x)

    with config.option_context('verbose', True):
        with config.option_context('verbose_log', logger):
            con.table('functional_alltypes', database=db.name)

    expected = 'DESC {0}.`functional_alltypes`'.format(db.name)

    assert len(queries) == 1
    assert queries[0] == expected


def test_sql_query_limits(alltypes):
    table = alltypes
    with config.option_context('sql.default_limit', 100000):
        # table has 25 rows
        assert len(table.execute()) == 7300
        # comply with limit arg for TableExpr
        assert len(table.execute(limit=10)) == 10
        # state hasn't changed
        assert len(table.execute()) == 7300
        # non-TableExpr ignores default_limit
        assert table.count().execute() == 7300
        # non-TableExpr doesn't observe limit arg
        assert table.count().execute(limit=10) == 7300
    with config.option_context('sql.default_limit', 20):
        # TableExpr observes default limit setting
        assert len(table.execute()) == 20
        # explicit limit= overrides default
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=23)) == 23
        # non-TableExpr ignores default_limit
        assert table.count().execute() == 7300
        # non-TableExpr doesn't observe limit arg
        assert table.count().execute(limit=10) == 7300
    # eliminating default_limit doesn't break anything
    with config.option_context('sql.default_limit', None):
        assert len(table.execute()) == 7300
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=10000)) == 7300
        assert table.count().execute() == 7300
        assert table.count().execute(limit=10) == 7300


def test_expr_compile_verify(alltypes):
    expr = alltypes.double_col.sum()

    assert isinstance(expr.compile(), str)
    assert expr.verify()


def test_api_compile_verify(alltypes):
    t = alltypes.timestamp_col

    supported = t.year()
    unsupported = t.rank()

    assert ibis.clickhouse.verify(supported)
    assert not ibis.clickhouse.verify(unsupported)


def test_database_repr(db):
    assert db.name in repr(db)


def test_database_default_current_database(con):
    db = con.database()
    assert db.name == con.current_database


def test_embedded_identifier_quoting(alltypes):
    t = alltypes

    expr = (t[[(t.double_col * 2).name('double(fun)')]]
            ['double(fun)'].sum())
    expr.execute()


def test_table_info(alltypes):
    buf = StringIO()
    alltypes.info(buf=buf)

    assert buf.getvalue() is not None


def test_execute_exprs_no_table_ref(con):
    cases = [
        (L(1) + L(2), 3)
    ]

    for expr, expected in cases:
        result = con.execute(expr)
        assert result == expected

    # ExprList
    exlist = ibis.api.expr_list([L(1).name('a'),
                                 ibis.now().name('b'),
                                 L(2).log().name('c')])
    con.execute(exlist)


def test_insert(con, alltypes, df):
    drop = 'DROP TABLE IF EXISTS temporary_alltypes'
    create = ('CREATE TABLE IF NOT EXISTS '
              'temporary_alltypes AS functional_alltypes')

    con.raw_sql(drop)
    con.raw_sql(create)

    temporary = con.table('temporary_alltypes')
    records = df[:10]

    assert len(temporary.execute()) == 0
    temporary.insert(records)

    tm.assert_frame_equal(temporary.execute(), records)


def test_insert_with_less_columns(con, alltypes, df):
    drop = 'DROP TABLE IF EXISTS temporary_alltypes'
    create = ('CREATE TABLE IF NOT EXISTS '
              'temporary_alltypes AS functional_alltypes')

    con.raw_sql(drop)
    con.raw_sql(create)

    temporary = con.table('temporary_alltypes')
    records = df.loc[:10, ['string_col']].copy()
    records['date_col'] = None

    with pytest.raises(AssertionError):
        temporary.insert(records)


def test_insert_with_more_columns(con, alltypes, df):
    drop = 'DROP TABLE IF EXISTS temporary_alltypes'
    create = ('CREATE TABLE IF NOT EXISTS '
              'temporary_alltypes AS functional_alltypes')

    con.raw_sql(drop)
    con.raw_sql(create)

    temporary = con.table('temporary_alltypes')
    records = df[:10].copy()
    records['non_existing_column'] = 'raise on me'

    with pytest.raises(AssertionError):
        temporary.insert(records)
