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

import os
import pytest
import unittest

import pandas as pd

import ibis.config as config
import ibis.connection as cnx
import ibis.expr.api as api
import ibis.expr.types as ir


class IbisTestEnv(object):

    def __init__(self):
        self.host = os.environ.get('IBIS_TEST_HOST', 'localhost')
        self.protocol = os.environ.get('IBIS_TEST_PROTOCOL', 'hiveserver2')
        self.port = os.environ.get('IBIS_TEST_PORT', 21050)


ENV = IbisTestEnv()


def connect(env):
    return cnx.impala_connect(host=ENV.host, protocol=ENV.protocol,
                              port=ENV.port)


pytestmark = pytest.mark.e2e


class ImpalaE2E(object):

    @classmethod
    def setUpClass(cls):
        try:
            import impala
            cls.con = connect(ENV)
        except ImportError:
            # fail gracefully if impyla not installed
            pytest.skip('no impyla')
        except Exception as e:
            if 'could not connect' in e.message.lower():
                pytest.skip('impalad not running')

    @classmethod
    def tearDownClass(cls):
        pass



class TestImpalaConnection(ImpalaE2E, unittest.TestCase):

    def test_get_table_ref(self):
        table = self.con.table('functional.alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_run_sql(self):
        query = """SELECT li.*
FROM tpch.lineitem li
  INNER JOIN tpch.orders o
    ON li.l_orderkey = o.o_orderkey
"""
        table = self.con.sql(query)

        li = self.con.table('tpch.lineitem')
        assert isinstance(table, ir.TableExpr)
        assert table.schema().equals(li.schema())

        expr = table.limit(10)
        result = expr.execute()
        assert len(result) == 10

    def test_result_as_dataframe(self):
        expr = self.con.table('functional.alltypes').limit(10)

        ex_names = expr.schema().names
        result = self.con.execute(expr)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ex_names
        assert len(result) == 10

    def test_adapt_scalar_array_results(self):
        table = self.con.table('functional.alltypes')

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

    def test_distinct_array(self):
        table = self.con.table('functional.alltypes')

        expr = table.string_col.distinct()
        result = self.con.execute(expr)
        assert isinstance(result, pd.Series)

    def test_decimal_metadata(self):
        table = self.con.table('tpch.lineitem')

        expr = table.l_quantity
        assert expr._precision == 12
        assert expr._scale == 2

        # TODO: what if user impyla version does not have decimal Metadata?

    def test_ctas_from_table_expr(self):
        expr = self.con.table('functional.alltypes')
        table_name = _random_table_name()

        try:
            self.con.create_table(table_name, expr, database='functional')
        except Exception:
            raise
        finally:
            _ensure_drop(self.con, table_name, database='functional')

    def test_insert_table(self):
        expr = self.con.table('functional.alltypes')
        table_name = _random_table_name()
        db = 'functional'

        try:
            self.con.create_table(table_name, expr.limit(0),
                                  database=db)
            self.con.insert(table_name, expr.limit(10), database=db)
            self.con.insert(table_name, expr.limit(10), database=db)

            sz = self.con.table('functional.{}'.format(table_name)).count()
            assert sz.execute() == 20

            # Overwrite and verify only 10 rows now
            self.con.insert(table_name, expr.limit(10), database=db,
                            overwrite=True)
            assert sz.execute() == 10
        except Exception:
            raise
        finally:
            _ensure_drop(self.con, table_name, database='functional')

    def test_builtins_1(self):
        table = self.con.table('functional.alltypes')

        i1 = table.tinyint_col
        i4 = table.int_col
        d = table.double_col
        s = table.string_col

        exprs = [
            api.now(),
            api.e,

            i4.zeroifnull(),

            d.abs(),
            d.cast('decimal(12, 2)'),
            d.cast('int32'),
            d.ceil(),
            d.exp(),
            d.isnull(),
            d.fillna(0),
            d.floor(),
            d.log(),
            d.ln(),
            d.log2(),
            d.log10(),
            d.notnull(),
            d.round(),
            d.round(2),
            d.sign(),
            d.sqrt(),
            d.zeroifnull(),

            # nullif cases
            5 / i1.nullif(0),
            5 / i1.nullif(i4),
            5 / i4.nullif(0),
            5 / d.nullif(0),

            api.literal(5).isin([i1, i4, d]),

            # coalesce-like cases
            api.coalesce(table.int_col,
                         api.null(),
                         table.smallint_col,
                         table.bigint_col, 5),
            api.greatest(table.float_col,
                         table.double_col, 5),
            api.least(table.string_col, 'foo'),

            # string stuff
            s.like('6%'),
            s.re_search('[\d]+'),
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs].limit(10)
        projection.execute()

    def test_aggregations_e2e(self):
        table = self.con.table('functional.alltypes').limit(100)

        d = table.double_col
        s = table.string_col

        exprs = [
            table.bool_col.count(),
            d.sum(),
            d.mean(),
            d.min(),
            d.max(),
            s.approx_nunique(),
            d.approx_median(),
            s.group_concat()
        ]

        agg_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()

    def test_tpch_self_join_failure(self):
        region = self.con.table('tpch.region')
        nation = self.con.table('tpch.nation')
        customer = self.con.table('tpch.customer')
        orders = self.con.table('tpch.orders')

        fields_of_interest = [
            region.r_name.name('region'),
            nation.n_name.name('nation'),
            orders.o_totalprice.name('amount'),
            orders.o_orderdate.cast('timestamp').name('odate')]

        joined_all = (
            region.join(nation, region.r_regionkey == nation.n_regionkey)
            .join(customer, customer.c_nationkey == nation.n_nationkey)
            .join(orders, orders.o_custkey == customer.c_custkey)
            [fields_of_interest])

        year = joined_all.odate.year().name('year')
        total = joined_all.amount.sum().cast('double').name('total')
        annual_amounts = (joined_all
                          .group_by(['region', year])
                          .aggregate(total))

        current = annual_amounts
        prior = annual_amounts.view()

        yoy_change = (current.total - prior.total).name('yoy_change')
        yoy = (current.join(prior, ((current.region == prior.region) &
                                    (current.year == (prior.year - 1))))
               [current.region, current.year, yoy_change])

        # it works!
        yoy.execute()

    def test_tpch_correlated_subquery_failure(self):
        # #183 and other issues
        region = self.con.table('tpch.region')
        nation = self.con.table('tpch.nation')
        customer = self.con.table('tpch.customer')
        orders = self.con.table('tpch.orders')

        fields_of_interest = [customer,
                              region.r_name.name('region'),
                              orders.o_totalprice.name('amount'),
                              orders.o_orderdate.cast('timestamp').name('odate')]

        tpch = (region.join(nation, region.r_regionkey == nation.n_regionkey)
                .join(customer, customer.c_nationkey == nation.n_nationkey)
                .join(orders, orders.o_custkey == customer.c_custkey)
                [fields_of_interest])

        t2 = tpch.view()
        conditional_avg = t2[(t2.region == tpch.region)].amount.mean()
        amount_filter = tpch.amount > conditional_avg

        expr = tpch[amount_filter].limit(0)
        expr.execute()

def _ensure_drop(con, table_name, database=None):
    con.drop_table(table_name, database=database,
                   must_exist=False)
    _assert_table_not_exists(con, table_name, database=database)

def _assert_table_not_exists(con, table_name, database=None):
    from impala.error import Error as ImpylaError

    if database is not None:
        tname = '.'.join((database, table_name))
    else:
        tname = table_name

    try:
        con.table(tname)
    except ImpylaError:
        pass
    except:
        raise


def _random_table_name():
    import uuid
    table_name = 'testing_' + uuid.uuid4().get_hex()
    return table_name


class TestQueryHDFSData(ImpalaE2E):

    def test_query_parquet_file(self):
        raise unittest.SkipTest

        hdfs_path = '/test-warehouse/functional_parquet.db/alltypesinsert'
        table = self.con.parquet_file(hdfs_path)

        name = table.op().name
        assert name.startswith('ibis_tmp_')

        # table exists
        self.con.table(name)

        expr = table.string_col.value_counts()
        expr.execute()

    def test_query_text_file_regex(self):
        pass

    def test_delimited_ascii(self):
        pass

    def test_avro(self):
        pass

    def test_cleanup_tmp_table_on_gc(self):
        # try:
        #     table.op().cleanup()
        # finally:
        #     _ensure_drop(table
        pass
