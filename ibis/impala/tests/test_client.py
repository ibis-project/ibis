# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

from ibis.compat import unittest
from ibis.impala.tests.common import IbisTestEnv, ImpalaE2E, connect_test
from ibis.tests.util import assert_equal
import ibis

import ibis.common as com
import ibis.config as config
import ibis.expr.types as ir
import ibis.util as util


ENV = IbisTestEnv()


class TestImpalaClient(ImpalaE2E, unittest.TestCase):

    def test_execute_exprs_default_backend(self):
        cases = [
            (ibis.literal(2), 2)
        ]

        ibis.options.default_backend = None
        client = connect_test(ENV, with_hdfs=False)
        assert ibis.options.default_backend is client

        for expr, expected in cases:
            result = expr.execute()
            assert result == expected

    def test_cursor_garbage_collection(self):
        for i in range(5):
            self.con.raw_sql('select 1', True).fetchall()
            self.con.raw_sql('select 1', True).fetchone()

    def test_raise_ibis_error_no_hdfs(self):
        # #299
        client = connect_test(ENV, with_hdfs=False)
        self.assertRaises(com.IbisError, getattr, client, 'hdfs')

    def test_get_table_ref(self):
        table = self.db.functional_alltypes
        assert isinstance(table, ir.TableExpr)

        table = self.db['functional_alltypes']
        assert isinstance(table, ir.TableExpr)

    def test_run_sql(self):
        query = """SELECT li.*
FROM {0}.tpch_lineitem li
""".format(self.test_data_db)
        table = self.con.sql(query)

        li = self.con.table('tpch_lineitem')
        assert isinstance(table, ir.TableExpr)
        assert_equal(table.schema(), li.schema())

        expr = table.limit(10)
        result = expr.execute()
        assert len(result) == 10

    def test_sql_with_limit(self):
        query = """\
SELECT *
FROM functional_alltypes
LIMIT 10"""
        table = self.con.sql(query)
        ex_schema = self.con.get_schema('functional_alltypes')
        assert_equal(table.schema(), ex_schema)

    def test_raw_sql(self):
        query = 'SELECT * from functional_alltypes limit 10'
        cur = self.con.raw_sql(query, results=True)
        rows = cur.fetchall()
        cur.release()
        assert len(rows) == 10

    def test_explain(self):
        t = self.con.table('functional_alltypes')
        expr = t.group_by('string_col').size()
        result = self.con.explain(expr)
        assert isinstance(result, str)

    def test_get_schema(self):
        t = self.con.table('tpch_lineitem')
        schema = self.con.get_schema('tpch_lineitem',
                                     database=self.test_data_db)
        assert_equal(t.schema(), schema)

    def test_result_as_dataframe(self):
        expr = self.alltypes.limit(10)

        ex_names = expr.schema().names
        result = self.con.execute(expr)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ex_names
        assert len(result) == 10

    def test_adapt_scalar_array_results(self):
        table = self.alltypes

        expr = table.double_col.sum()
        result = self.con.execute(expr)
        assert isinstance(result, float)

        with config.option_context('interactive', True):
            result2 = expr.execute()
            assert isinstance(result2, float)

        expr = (table.group_by('string_col')
                .aggregate([table.count().name('count')])
                .string_col)

        result = self.con.execute(expr)
        assert isinstance(result, pd.Series)

    def test_interactive_repr_call_failure(self):
        t = self.con.table('tpch_lineitem').limit(100000)

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

    def test_array_default_limit(self):
        t = self.alltypes

        result = self.con.execute(t.float_col, limit=100)
        assert len(result) == 100

    def test_limit_overrides_expr(self):
        # #418
        t = self.alltypes
        result = self.con.execute(t.limit(10), limit=5)
        assert len(result) == 5

    def test_limit_equals_none_no_limit(self):
        t = self.alltypes

        with config.option_context('sql.default_limit', 10):
            result = t.execute(limit=None)
            assert len(result) > 10

    def test_verbose_log_queries(self):
        queries = []

        def logger(x):
            queries.append(x)

        with config.option_context('verbose', True):
            with config.option_context('verbose_log', logger):
                self.con.table('tpch_orders', database=self.test_data_db)

        assert len(queries) == 1
        expected = 'DESCRIBE {0}.`tpch_orders`'.format(self.test_data_db)
        assert queries[0] == expected

    def test_sql_query_limits(self):
        table = self.con.table('tpch_nation', database=self.test_data_db)
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

    def test_expr_compile_verify(self):
        table = self.db.functional_alltypes
        expr = table.double_col.sum()

        assert isinstance(expr.compile(), str)
        assert expr.verify()

    def test_api_compile_verify(self):
        t = self.db.functional_alltypes

        s = t.string_col

        supported = s.lower()
        unsupported = s.replace('foo', 'bar')

        assert ibis.impala.verify(supported)
        assert not ibis.impala.verify(unsupported)

    def test_database_repr(self):
        assert self.test_data_db in repr(self.db)

    def test_database_drop(self):
        tmp_name = '__ibis_test_{0}'.format(util.guid())
        self.con.create_database(tmp_name)

        db = self.con.database(tmp_name)
        self.temp_databases.append(tmp_name)
        db.drop()
        assert not self.con.exists_database(tmp_name)

    def test_database_default_current_database(self):
        db = self.con.database()
        assert db.name == self.con.current_database

    def test_namespace(self):
        ns = self.db.namespace('tpch_')

        assert 'tpch_' in repr(ns)

        table = ns.lineitem
        expected = self.db.tpch_lineitem
        attrs = dir(ns)
        assert 'lineitem' in attrs
        assert 'functional_alltypes' not in attrs

        assert_equal(table, expected)

    def test_close_drops_temp_tables(self):
        from posixpath import join as pjoin

        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

        client = connect_test(ENV)
        table = client.parquet_file(hdfs_path)

        name = table.op().name
        assert self.con.exists_table(name) is True
        client.close()

        assert not self.con.exists_table(name)

    def test_execute_async_simple(self):
        t = self.db.functional_alltypes
        expr = t.double_col.sum()

        q = expr.execute(async=True)
        result = q.get_result()
        expected = expr.execute()
        assert result == expected

    def test_query_cancel(self):
        import time
        t = self.db.functional_alltypes

        t2 = t.union(t).union(t)

        # WM: this query takes about 90 seconds to execute for me locally, so
        # I'm eyeballing an acceptable time frame for the cancel to work
        expr = t2.join(t2).count()

        start = time.clock()
        q = expr.execute(async=True)
        q.cancel()
        end = time.clock()
        elapsed = end - start
        assert elapsed < 5

        assert q.is_finished()

    def test_set_compression_codec(self):
        old_opts = self.con.get_options()
        assert old_opts['COMPRESSION_CODEC'].upper() == 'NONE'

        self.con.set_compression_codec('snappy')
        opts = self.con.get_options()
        assert opts['COMPRESSION_CODEC'].upper() == 'SNAPPY'

        self.con.set_compression_codec(None)
        opts = self.con.get_options()
        assert opts['COMPRESSION_CODEC'].upper() == 'NONE'

    def test_disable_codegen(self):
        self.con.disable_codegen(False)
        opts = self.con.get_options()
        assert opts['DISABLE_CODEGEN'] == '0'

        self.con.disable_codegen()
        opts = self.con.get_options()
        assert opts['DISABLE_CODEGEN'] == '1'

        impala_con = self.con.con
        cur1 = impala_con.execute('SET')
        cur2 = impala_con.execute('SET')

        opts1 = dict(cur1.fetchall())
        cur1.release()

        opts2 = dict(cur2.fetchall())
        cur2.release()

        assert opts1['DISABLE_CODEGEN'] == '1'
        assert opts2['DISABLE_CODEGEN'] == '1'

    def test_attr_name_conflict(self):
        LEFT = 'testing_{0}'.format(util.guid())
        RIGHT = 'testing_{0}'.format(util.guid())

        schema = ibis.schema([('id', 'int32'), ('name', 'string'),
                              ('files', 'int32')])

        db = self.con.database(self.tmp_db)

        for tablename in (LEFT, RIGHT):
            db.create_table(tablename, schema=schema,
                            format='parquet')

        left = db[LEFT]
        right = db[RIGHT]

        left.join(right, ['id'])
        left.join(right, ['id', 'name'])
        left.join(right, ['id', 'files'])
