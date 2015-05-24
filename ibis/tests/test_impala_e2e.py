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

import gc
import os
import pytest
import unittest

import pandas as pd

from hdfs import InsecureClient

import ibis

import ibis.config as config
import ibis.expr.api as api
import ibis.expr.types as ir


class IbisTestEnv(object):

    def __init__(self):
        self.host = os.environ.get('IBIS_TEST_HOST', 'localhost')
        self.protocol = os.environ.get('IBIS_TEST_PROTOCOL', 'hiveserver2')
        self.port = os.environ.get('IBIS_TEST_PORT', 21050)

        # Impala dev environment uses port 5070 for HDFS web interface

        hdfs_host = 'localhost'
        webhdfs_port = 5070
        url = 'http://{}:{}'.format(hdfs_host, webhdfs_port)
        self.hdfs = InsecureClient(url)


ENV = IbisTestEnv()


def connect(env):
    con = ibis.impala_connect(host=ENV.host,
                              protocol=ENV.protocol,
                              port=ENV.port)
    return ibis.make_client(con, ENV.hdfs)


pytestmark = pytest.mark.e2e


class ImpalaE2E(object):

    @classmethod
    def setUpClass(cls):
        try:
            import impala  # noqa
            cls.con = connect(ENV)
        except ImportError:
            # fail gracefully if impyla not installed
            pytest.skip('no impyla')
        except Exception as e:
            if 'could not connect' in e.message.lower():
                pytest.skip('impalad not running')
            raise

    @classmethod
    def tearDownClass(cls):
        pass


class TestImpalaConnection(ImpalaE2E, unittest.TestCase):

    def test_get_table_ref(self):
        table = self.con.table('functional.alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_table_name_reserved_identifier(self):
        table_name = 'distinct'
        expr = self.con.table('functional.alltypes')
        self.con.create_table(table_name, expr)

        t = self.con.table(table_name)
        t.limit(10).execute()

        _ensure_drop(self.con, 'distinct')

    def test_list_databases(self):
        assert len(self.con.list_databases()) > 0

    def test_list_tables(self):
        assert len(self.con.list_tables(database='tpch')) > 0
        assert len(self.con.list_tables(like='nat*', database='tpch')) > 0

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

            # modulus cases
            i1 % 5,
            i4 % 10,
            20 % i1,
            d % 5,

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
            s.contains('6'),
            s.like('6%'),
            s.re_search('[\d]+'),
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs].limit(10)
        projection.execute()

    def test_decimal_timestamp_builtins(self):
        table = self.con.table('tpch.lineitem')

        dc = table.l_quantity
        ts = table.l_receiptdate.cast('timestamp')

        exprs = [
            dc % 10,
            dc + 5,
            dc + dc,
            dc / 2,
            dc * 2,
            dc ** 2,
            dc.cast('double'),

            api.where(table.l_discount > 0,
                      dc * table.l_discount, api.NA),

            dc.fillna(0),
        ]

        timestamp_fields = ['year', 'month', 'day', 'hour', 'minute',
                            'second', 'millisecond', 'microsecond',
                            'nanosecond', 'week']
        for field in timestamp_fields:
            if hasattr(ts, field):
                exprs.append(getattr(ts, field)())

            offset = getattr(ibis, field)(2)
            exprs.append(ts + offset)
            exprs.append(ts - offset)

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
        region = self.con.table('region', database='tpch_parquet')
        nation = self.con.table('nation', database='tpch')
        customer = self.con.table('customer', database='tpch')
        orders = self.con.table('orders', database='tpch_parquet')

        fields_of_interest = [customer,
                              region.r_name.name('region'),
                              orders.o_totalprice.name('amount'),
                              orders.o_orderdate
                              .cast('timestamp').name('odate')]

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


class TestQueryHDFSData(ImpalaE2E, unittest.TestCase):

    def test_cleanup_tmp_table_on_gc(self):
        hdfs_path = '/test-warehouse/tpch.region_parquet'
        table = self.con.parquet_file(hdfs_path)
        name = table.op().name
        table = None
        gc.collect()
        _assert_table_not_exists(self.con, name)

    def test_persist_parquet_file_with_name(self):
        hdfs_path = '/test-warehouse/tpch.region_parquet'

        name = _random_table_name()
        schema = ibis.schema([('r_regionkey', 'int16'),
                              ('r_name', 'string'),
                              ('r_comment', 'string')])
        self.con.parquet_file(hdfs_path, schema=schema,
                              name=name, persist=True)
        gc.collect()

        # table still exists
        self.con.table(name)

        _ensure_drop(self.con, name)

    def test_query_avro(self):
        hdfs_path = '/test-warehouse/tpch.region_avro'
        schema = ibis.schema([('r_regionkey', 'int16'),
                              ('r_name', 'string'),
                              ('r_comment', 'string')])

        avro_schema = {
            "fields": [
                {"type": ["int", "null"], "name": "R_REGIONKEY"},
                {"type": ["string", "null"], "name": "R_NAME"},
                {"type": ["string", "null"], "name": "R_COMMENT"}],
            "type": "record",
            "name": "a"
        }

        table = self.con.avro_file(hdfs_path, schema, avro_schema)

        name = table.op().name
        assert name.startswith('ibis_tmp_')

        # table exists
        self.con.table(name)

        expr = table.r_name.value_counts()
        expr.execute()

        assert table.count().execute() == 5

    def test_query_parquet_file_with_schema(self):
        hdfs_path = '/test-warehouse/tpch.region_parquet'
        schema = ibis.schema([('r_regionkey', 'int16'),
                              ('r_name', 'string'),
                              ('r_comment', 'string')])

        table = self.con.parquet_file(hdfs_path, schema=schema)

        name = table.op().name
        assert name.startswith('ibis_tmp_')

        # table exists
        self.con.table(name)

        expr = table.r_name.value_counts()
        expr.execute()

        assert table.count().execute() == 5

    def test_query_parquet_file_like_table(self):
        hdfs_path = '/test-warehouse/tpch.region_parquet'

        ex_schema = ibis.schema([('r_regionkey', 'int16'),
                                 ('r_name', 'string'),
                                 ('r_comment', 'string')])

        table = self.con.parquet_file(hdfs_path, like_table='tpch.region')

        assert table.schema().equals(ex_schema)

    def test_query_parquet_infer_schema(self):
        hdfs_path = '/test-warehouse/tpch.region_parquet'
        table = self.con.parquet_file(hdfs_path)

        ex_schema = ibis.schema([('r_regionkey', 'int32'),
                                 ('r_name', 'string'),
                                 ('r_comment', 'string')])

        assert table.schema().equals(ex_schema)

    def test_query_text_file_regex(self):
        pass

    def test_query_delimited_file_directory(self):
        hdfs_path = '/ibis-test/csv-test'

        schema = ibis.schema([('foo', 'string'),
                              ('bar', 'double'),
                              ('baz', 'int8')])
        table = self.con.delimited_file(hdfs_path, schema, delimiter=',')

        expr = (table
                [table.bar > 0]
                .group_by('foo')
                .aggregate([table.bar.sum().name('sum(bar)'),
                            table.baz.sum().name('mean(baz)')]))
        expr.execute()

    def test_avro(self):
        pass
