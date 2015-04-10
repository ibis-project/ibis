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


class TestImpalaConnection(unittest.TestCase):

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
        assert expr.precision == 12
        assert expr.scale == 2

        # TODO: what if user impyla version does not have decimal Metadata?

    def test_ctas_from_table_expr(self):
        import uuid
        table_name = 'testing_' + uuid.uuid4().get_hex()

        expr = self.con.table('functional.alltypes')

        try:
            self.con.create_table(table_name, expr, database='functional')
        except Exception:
            raise
        finally:
            self._ensure_drop(table_name, database='functional')

    def test_builtins_1(self):
        table = self.con.table('functional.alltypes')

        i1 = table.tinyint_col
        i4 = table.int_col
        d = table.double_col

        exprs = [

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

            # coalesce-like cases
            api.coalesce(table.int_col,
                         api.null(),
                         table.smallint_col,
                         table.bigint_col, 5),
            api.greatest(table.float_col,
                         table.double_col, 5),
            api.least(table.string_col, 'foo'),
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs].limit(10)
        projection.execute()

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

    def _ensure_drop(self, table_name, database=None):
        self.con.drop_table(table_name, database=database,
                            must_exist=False)
        self._assert_table_not_exists(table_name, database=database)

    def _assert_table_not_exists(self, table_name, database=None):
        from impala.error import Error as ImpylaError

        if database is not None:
            tname = '.'.join((database, table_name))
        else:
            tname = table_name

        try:
            self.con.table(tname)
        except ImpylaError:
            pass
        except:
            raise
