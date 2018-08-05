import datetime
import time
import pandas as pd
import pytz
import pytest

import ibis
import ibis.common as com
import ibis.config as config
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.util as util

from ibis.tests.util import assert_equal

pytest.importorskip('sqlalchemy')
pytest.importorskip('hdfs')
pytest.importorskip('impala.dbapi')

pytestmark = pytest.mark.impala


@pytest.fixture(scope='module')
def db(con, test_data_db):
    return con.database(test_data_db)


def test_execute_exprs_default_backend(con_no_hdfs):
    expr = ibis.literal(2)
    expected = 2

    assert ibis.options.default_backend is not None

    result = expr.execute()
    assert result == expected


def test_cursor_garbage_collection(con):
    for i in range(5):
        con.raw_sql('select 1', True).fetchall()
        con.raw_sql('select 1', True).fetchone()


def test_raise_ibis_error_no_hdfs(con_no_hdfs):
    # GH299
    with pytest.raises(com.IbisError):
        con_no_hdfs.hdfs


def test_get_table_ref(db):
    assert isinstance(db.functional_alltypes, ir.TableExpr)
    assert isinstance(db['functional_alltypes'], ir.TableExpr)


def test_run_sql(con, test_data_db):
    query = """SELECT li.*
FROM {0}.tpch_lineitem li
""".format(test_data_db)
    table = con.sql(query)

    li = con.table('tpch_lineitem')
    assert isinstance(table, ir.TableExpr)
    assert_equal(table.schema(), li.schema())

    expr = table.limit(10)
    result = expr.execute()
    assert len(result) == 10


def test_sql_with_limit(con):
    query = """\
SELECT *
FROM functional_alltypes
LIMIT 10"""
    table = con.sql(query)
    ex_schema = con.get_schema('functional_alltypes')
    assert_equal(table.schema(), ex_schema)


def test_raw_sql(con):
    query = 'SELECT * from functional_alltypes limit 10'
    cur = con.raw_sql(query, results=True)
    rows = cur.fetchall()
    cur.release()
    assert len(rows) == 10


def test_explain(con):
    t = con.table('functional_alltypes')
    expr = t.group_by('string_col').size()
    result = con.explain(expr)
    assert isinstance(result, str)


def test_get_schema(con, test_data_db):
    t = con.table('tpch_lineitem')
    schema = con.get_schema('tpch_lineitem', database=test_data_db)
    assert_equal(t.schema(), schema)


def test_result_as_dataframe(con, alltypes):
    expr = alltypes.limit(10)

    ex_names = expr.schema().names
    result = con.execute(expr)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ex_names
    assert len(result) == 10


def test_adapt_scalar_array_results(con, alltypes):
    table = alltypes

    expr = table.double_col.sum()
    result = con.execute(expr)
    assert isinstance(result, float)

    with config.option_context('interactive', True):
        result2 = expr.execute()
        assert isinstance(result2, float)

    expr = (table.group_by('string_col')
            .aggregate([table.count().name('count')])
            .string_col)

    result = con.execute(expr)
    assert isinstance(result, pd.Series)


def test_interactive_repr_call_failure(con):
    t = con.table('tpch_lineitem').limit(100000)

    t = t[t, t.l_receiptdate.cast('timestamp').name('date')]

    keys = [t.date.year().name('year'), 'l_linestatus']
    filt = t.l_linestatus.isin(['F'])
    expr = (t[filt]
            .group_by(keys)
            .aggregate(t.l_extendedprice.mean().name('avg_px')))

    w2 = ibis.trailing_window(9, group_by=expr.l_linestatus,
                              order_by=expr.year)

    metric = expr['avg_px'].mean().over(w2)
    enriched = expr[expr, metric]
    with config.option_context('interactive', True):
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

    with config.option_context('sql.default_limit', 10):
        result = t.execute(limit=None)
        assert len(result) > 10


def test_verbose_log_queries(con, test_data_db):
    queries = []

    with config.option_context('verbose', True):
        with config.option_context('verbose_log', queries.append):
            con.table('tpch_orders', database=test_data_db)

    assert len(queries) == 1
    query, = queries
    expected = 'DESCRIBE {}.`tpch_orders`'.format(test_data_db)
    assert query == expected


def test_sql_query_limits(con, test_data_db):
    table = con.table('tpch_nation', database=test_data_db)
    with config.option_context('sql.default_limit', 100000):
        # table has 25 rows
        assert len(table.execute()) == 25
        # comply with limit arg for TableExpr
        assert len(table.execute(limit=10)) == 10
        # state hasn't changed
        assert len(table.execute()) == 25
        # non-TableExpr ignores default_limit
        assert table.count().execute() == 25
        # non-TableExpr doesn't observe limit arg
        assert table.count().execute(limit=10) == 25
    with config.option_context('sql.default_limit', 20):
        # TableExpr observes default limit setting
        assert len(table.execute()) == 20
        # explicit limit= overrides default
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=23)) == 23
        # non-TableExpr ignores default_limit
        assert table.count().execute() == 25
        # non-TableExpr doesn't observe limit arg
        assert table.count().execute(limit=10) == 25
    # eliminating default_limit doesn't break anything
    with config.option_context('sql.default_limit', None):
        assert len(table.execute()) == 25
        assert len(table.execute(limit=15)) == 15
        assert len(table.execute(limit=10000)) == 25
        assert table.count().execute() == 25
        assert table.count().execute(limit=10) == 25


def test_expr_compile_verify(db):
    table = db.functional_alltypes
    expr = table.double_col.sum()

    assert isinstance(expr.compile(), str)
    assert expr.verify()


def test_api_compile_verify(db):
    t = db.functional_alltypes

    s = t.string_col

    supported = s.lower()
    unsupported = s.replace('foo', 'bar')

    assert ibis.impala.verify(supported)
    assert not ibis.impala.verify(unsupported)


def test_database_repr(db, test_data_db):
    assert test_data_db in repr(db)


def test_database_default_current_database(con):
    db = con.database()
    assert db.name == con.current_database


def test_namespace(db):
    ns = db.namespace('tpch_')

    assert 'tpch_' in repr(ns)

    table = ns.lineitem
    expected = db.tpch_lineitem
    attrs = dir(ns)
    assert 'lineitem' in attrs
    assert 'functional_alltypes' not in attrs

    assert_equal(table, expected)


def test_close_drops_temp_tables(con, test_data_dir):
    from posixpath import join as pjoin

    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    table = con.parquet_file(hdfs_path)

    name = table.op().name
    assert con.exists_table(name) is True
    con.close()

    assert not con.exists_table(name)


def test_set_compression_codec(con):
    old_opts = con.get_options()
    assert old_opts['COMPRESSION_CODEC'].upper() in ('NONE', '')

    con.set_compression_codec('snappy')
    opts = con.get_options()
    assert opts['COMPRESSION_CODEC'].upper() == 'SNAPPY'

    con.set_compression_codec(None)
    opts = con.get_options()
    assert opts['COMPRESSION_CODEC'].upper() in ('NONE', '')


def test_disable_codegen(con):
    con.disable_codegen(False)
    opts = con.get_options()
    assert opts['DISABLE_CODEGEN'] == '0'

    con.disable_codegen()
    opts = con.get_options()
    assert opts['DISABLE_CODEGEN'] == '1'

    impala_con = con.con
    cur1 = impala_con.execute('SET')
    cur2 = impala_con.execute('SET')

    opts1 = dict(row[:2] for row in cur1.fetchall())
    cur1.release()

    opts2 = dict(row[:2] for row in cur2.fetchall())
    cur2.release()

    assert opts1['DISABLE_CODEGEN'] == '1'
    assert opts2['DISABLE_CODEGEN'] == '1'


def test_attr_name_conflict(
    con, tmp_db, temp_parquet_table, temp_parquet_table2
):
    left = temp_parquet_table
    right = temp_parquet_table2

    assert left.join(right, ['id']) is not None
    assert left.join(right, ['id', 'name']) is not None
    assert left.join(right, ['id', 'files']) is not None


@pytest.fixture(scope='session')
def con2(env):
    con = ibis.impala.connect(host=env.impala_host,
                              port=env.impala_port,
                              auth_mechanism=env.auth_mechanism)
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


def test_rerelease_cursor(con2):
    # we use a separate `con2` fixture here because any connection pool
    # manipulation we want to happen independently of `con`
    with con2.raw_sql('select 1', True) as cur1:
        pass

    cur1.release()

    with con2.raw_sql('select 1', True) as cur2:
        pass

    cur2.release()

    with con2.raw_sql('select 1', True) as cur3:
        pass

    assert cur1 == cur2
    assert cur2 == cur3


def test_day_of_week(con):
    date_var = ibis.literal(datetime.date(2017, 1, 1), type=dt.date)
    expr_index = date_var.day_of_week.index()
    result = con.execute(expr_index)
    assert result == 6

    expr_name = date_var.day_of_week.full_name()
    result = con.execute(expr_name)
    assert result == 'Sunday'


def test_time_to_int_cast(con):
    now = pytz.utc.localize(datetime.datetime.now())
    d = ibis.literal(now)
    result = con.execute(d.cast('int64'))
    assert result == int(time.mktime(now.timetuple())) * 1000000


def test_set_option_with_dot(con):
    con.set_options({'request_pool': 'baz.quux'})
    result = dict(row[:2] for row in con.raw_sql('set', True).fetchall())
    assert result['REQUEST_POOL'] == 'baz.quux'


def test_list_databases(con):
    assert con.list_databases()


def test_list_tables(con, test_data_db):
    assert con.list_tables(database=test_data_db)
    assert con.list_tables(like='*nat*', database=test_data_db)


def test_set_database(con_no_db, test_data_db):
    # create new connection with no default db set
    # TODO: set test_data_db to None
    with pytest.raises(Exception):
        con_no_db.table('functional_alltypes')
    con_no_db.set_database(test_data_db)
    assert con_no_db.table('functional_alltypes') is not None


def test_tables_robust_to_set_database(con, test_data_db, temp_database):
    table = con.table('functional_alltypes', database=test_data_db)
    con.set_database(temp_database)
    assert con.current_database == temp_database

    # it still works!
    n = 10
    df = table.limit(n).execute()
    assert len(df) == n


def test_exists_table(con):
    assert con.exists_table('functional_alltypes')
    assert not con.exists_table('foobarbaz_{}'.format(util.guid()))


def text_exists_table_with_database(
    con, alltypes, test_data_db, temp_table, temp_database
):
    tmp_db = test_data_db
    con.create_table(temp_table, alltypes, database=tmp_db)

    assert con.exists_table(temp_table, database=tmp_db)
    assert not con.exists_table(temp_table, database=temp_database)
