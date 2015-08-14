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
from ibis.tests.util import IbisTestEnv, ImpalaE2E, assert_equal, connect_test

import ibis.common as com
import ibis.config as config
import ibis.expr.types as ir
import ibis.util as util


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


ENV = IbisTestEnv()


class TestImpalaClient(ImpalaE2E, unittest.TestCase):

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

    def test_array_default_limit(self):
        t = self.alltypes

        result = self.con.execute(t.float_col, limit=100)
        assert len(result) == 100

    def test_limit_overrides_expr(self):
        # #418
        t = self.alltypes
        result = self.con.execute(t.limit(10), limit=5)
        assert len(result) == 5

    def test_verbose_log_queries(self):
        queries = []

        def logger(x):
            queries.append(x)

        with config.option_context('verbose', True):
            with config.option_context('verbose_log', logger):
                self.con.table('tpch_orders', database=self.test_data_db)

        assert len(queries) == 1
        expected = 'SELECT * FROM {0}.`tpch_orders` LIMIT 0'.format(
            self.test_data_db)
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

    def test_database_repr(self):
        assert self.test_data_db in repr(self.db)

    def test_database_drop(self):
        tmp_name = '__ibis_test_{0}'.format(util.guid())
        self.con.create_database(tmp_name)

        db = self.con.database(tmp_name)
        self.temp_databases.append(tmp_name)
        db.drop()
        assert not self.con.exists_database(tmp_name)

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
