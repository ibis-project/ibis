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

from posixpath import join as pjoin
from copy import copy
import gc
import pytest

import pandas as pd
from decimal import Decimal

from hdfs import InsecureClient
import ibis

from ibis.compat import unittest
from ibis.sql.compiler import to_sql
from ibis.tests.util import IbisTestEnv

import ibis.common as com
import ibis.config as config
import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.util as util


ENV = IbisTestEnv()


def connect(env, with_hdfs=True):
    con = ibis.impala_connect(host=env.impala_host,
                              protocol=env.impala_protocol,
                              database=env.test_data_db,
                              port=env.impala_port)
    if with_hdfs:
        hdfs_client = InsecureClient(env.hdfs_url)
        return ibis.make_client(con, hdfs_client)
    else:
        return ibis.make_client(con)


pytestmark = pytest.mark.e2e


class ImpalaE2E(object):

    @classmethod
    def setUpClass(cls):
        cls.con = connect(ENV)
        # Tests run generally faster without it
        if not ENV.use_codegen:
            cls.con.disable_codegen()
        cls.hdfs = cls.con.hdfs
        cls.test_data_dir = ENV.test_data_dir
        cls.test_data_db = ENV.test_data_db
        cls.tmp_dir = ENV.tmp_dir
        cls.tmp_db = '__ibis_{0}'.format(util.guid())
        cls.con.create_database(cls.tmp_db)
        cls.alltypes = cls.con.table('functional_alltypes')

    @classmethod
    def tearDownClass(cls):
        cls.con.drop_database(cls.tmp_db, force=True)


class TestImpalaConnection(ImpalaE2E, unittest.TestCase):

    def test_raise_ibis_error_no_hdfs(self):
        # #299
        client = connect(ENV, with_hdfs=False)
        self.assertRaises(com.IbisError, getattr, client, 'hdfs')

    def test_get_table_ref(self):
        table = self.con.table('functional_alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_table_name_reserved_identifier(self):
        table_name = 'distinct'
        expr = self.con.table('functional_alltypes')
        self.con.create_table(table_name, expr)

        t = self.con.table(table_name)
        t.limit(10).execute()

        _ensure_drop(self.con, 'distinct')

    def test_list_databases(self):
        assert len(self.con.list_databases()) > 0

    def test_list_tables(self):
        assert len(self.con.list_tables(database=self.test_data_db)) > 0
        assert len(self.con.list_tables(like='*nat*',
                                        database=self.test_data_db)) > 0

    def test_set_database(self):
        # create new connection with no default db set
        env = copy(ENV)
        env.test_data_db = None
        con = connect(env)
        self.assertRaises(Exception, con.table, 'functional_alltypes')
        con.set_database(self.test_data_db)
        con.table('functional_alltypes')

    def test_create_exists_drop_database(self):
        tmp_name = util.guid()

        assert not self.con.exists_database(tmp_name)

        self.con.create_database(tmp_name)
        assert self.con.exists_database(tmp_name)

        self.con.drop_database(tmp_name)
        assert not self.con.exists_database(tmp_name)

    def test_drop_non_empty_database(self):
        tmp_db = util.guid()
        self.con.create_database(tmp_db)

        tmp_table = util.guid()
        self.con.create_table(tmp_table, self.alltypes, database=tmp_db)

        self.assertRaises(com.IntegrityError, self.con.drop_database, tmp_db)

        self.con.drop_database(tmp_db, force=True)
        assert not self.con.exists_database(tmp_db)

    def test_create_database_with_location(self):
        base = pjoin(self.tmp_dir, util.guid())
        name = 'test_{0}'.format(util.guid())
        tmp_path = pjoin(base, name)

        self.con.create_database(name, path=tmp_path)
        assert self.hdfs.exists(base)
        self.con.drop_database(name)
        self.hdfs.rmdir(base)

    def test_drop_table_not_exist(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_table, random_name)

        self.con.drop_table(random_name, force=True)

    def test_exists_table(self):
        pass

    def test_run_sql(self):
        query = """SELECT li.*
FROM ibis_testing.tpch_lineitem li
  INNER JOIN ibis_testing.tpch_orders o
    ON li.l_orderkey = o.o_orderkey
"""
        table = self.con.sql(query)

        li = self.con.table('tpch_lineitem')
        assert isinstance(table, ir.TableExpr)
        assert table.schema().equals(li.schema())

        expr = table.limit(10)
        result = expr.execute()
        assert len(result) == 10

    def test_get_schema(self):
        t = self.con.table('tpch_lineitem')
        schema = self.con.get_schema('tpch_lineitem', database='ibis_testing')
        assert t.schema().equals(schema)

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

    def test_execute_exprs_no_table_ref(self):
        cases = [
            (ibis.literal(1) + ibis.literal(2), 3)
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected

        # ExprList
        exlist = ibis.api.expr_list([ibis.literal(1).name('a'),
                                     ibis.now().name('b'),
                                     ibis.literal(2).log().name('c')])
        self.con.execute(exlist)

    def test_summary_execute(self):
        table = self.alltypes

        # also test set_column while we're at it
        table = table.set_column('double_col',
                                 table.double_col * 2)

        expr = table.double_col.summary()
        repr(expr)

        result = expr.execute()
        assert isinstance(result, pd.DataFrame)

        expr = (table.group_by('string_col')
                .aggregate([table.double_col.summary().prefix('double_'),
                            table.float_col.summary().prefix('float_'),
                            table.string_col.summary().suffix('_string')]))
        result = expr.execute()
        assert isinstance(result, pd.DataFrame)

    def test_distinct_array(self):
        table = self.alltypes

        expr = table.string_col.distinct()
        result = self.con.execute(expr)
        assert isinstance(result, pd.Series)

    def test_decimal_metadata(self):
        table = self.con.table('tpch_lineitem')

        expr = table.l_quantity
        assert expr._precision == 12
        assert expr._scale == 2

        # TODO: what if user impyla version does not have decimal Metadata?

    def test_ctas_from_table_expr(self):
        expr = self.alltypes
        table_name = _random_table_name()

        try:
            self.con.create_table(table_name, expr, database=self.test_data_db)
        except Exception:
            raise
        finally:
            _ensure_drop(self.con, table_name, database=self.test_data_db)

    def test_insert_table(self):
        expr = self.alltypes
        table_name = _random_table_name()
        db = self.test_data_db

        try:
            self.con.create_table(table_name, expr.limit(0),
                                  database=db)
            self.con.insert(table_name, expr.limit(10), database=db)
            self.con.insert(table_name, expr.limit(10), database=db)

            sz = self.con.table('{0}.{1}'.format(db, table_name)).count()
            assert sz.execute() == 20

            # Overwrite and verify only 10 rows now
            self.con.insert(table_name, expr.limit(10), database=db,
                            overwrite=True)
            assert sz.execute() == 10
        except Exception:
            raise
        finally:
            _ensure_drop(self.con, table_name, database=db)

    def test_builtins_1(self):
        table = self.alltypes

        i1 = table.tinyint_col
        i4 = table.int_col
        i8 = table.bigint_col
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

            i4.to_timestamp('s'),
            i4.to_timestamp('ms'),
            i4.to_timestamp('us'),

            i8.to_timestamp(),

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
            d.round(i1),

            d.sign(),
            d.sqrt(),
            d.zeroifnull(),

            # nullif cases
            5 / i1.nullif(0),
            5 / i1.nullif(i4),
            5 / i4.nullif(0),
            5 / d.nullif(0),

            api.literal(5).isin([i1, i4, d]),

            # tier and histogram
            d.bucket([0, 10, 25, 50, 100]),
            d.bucket([0, 10, 25, 50], include_over=True),
            d.bucket([0, 10, 25, 50], include_over=True, close_extreme=False),
            d.bucket([10, 25, 50, 100], include_under=True),

            d.histogram(10),
            d.histogram(5, base=10),
            d.histogram(base=10, binwidth=5),

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
            s.re_extract('[\d]+', 0),
            s.re_replace('[\d]+', 'a'),
            s.repeat(2),
            s.find("a"),
            s.translate("a", "b"),
            s.locate("a"),
            s.lpad(10, "a"),
            s.rpad(10, "a"),
            s.find_in_set(["a"]),
            s.lower(),
            s.upper(),
            s.reverse(),
            s.ascii_str(),
            s.length(),
            s.strip(),
            s.lstrip(),
            s.strip(),
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs].limit(10)
        projection.execute()

    def assert_cases_equality(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected, to_sql(expr)

    def test_int_builtins(self):
        i8 = ibis.literal(50)
        i32 = ibis.literal(50000)
        i64 = ibis.literal(5 * 10 ** 8)
        mod_cases = [
            (i8 % 5, 0),
            (i32 % 10, 0),
            (250 % i8, 0),
        ]
        nullif_cases = [
            (5 / i8.nullif(0), 0.1),
            (5 / i8.nullif(i32), 0.1),
            (5 / i32.nullif(0), 0.0001),
            (i32.zeroifnull(), 50000),
        ]
        timestamp_cases = [
            (i32.to_timestamp('s'), pd.to_datetime(50000, unit='s')),
            (i32.to_timestamp('ms'), pd.to_datetime(50000, unit='ms')),
            (i64.to_timestamp(), pd.to_datetime(5 * 10 ** 8, unit='s')),
        ]

        self.assert_cases_equality(mod_cases + nullif_cases + timestamp_cases)

    def test_decimal_builtins(self):
        d = ibis.literal(5.245)
        general_cases = [
            (ibis.literal(-5).abs(), 5),
            (d.cast('int32'), 5),
            (d.ceil(), 6),
            (d.isnull(), False),
            (d.floor(), 5),
            (d.notnull(), True),
            (d.round(), 5),
            (d.round(2), 5.25),
            (d.sign(), 1),
        ]
        self.assert_cases_equality(general_cases)

    def test_decimal_builtins_2(self):
        d = ibis.literal('5.245')
        dc = d.cast('decimal(12,5)')
        cases = [
            (dc % 5, Decimal('0.245')),

            (dc.fillna(0), Decimal('5.245')),

            (dc.exp(), 189.6158),
            (dc.log(), 1.65728),
            (dc.log2(), 2.39094),
            (dc.log10(), 0.71975),
            (dc.sqrt(), 2.29019),
            (dc.zeroifnull(), Decimal(5.245)),
            (-dc, Decimal('-5.245'))
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)

            def approx_equal(a, b, eps=0.0001):
                assert abs(a - b) < eps
            approx_equal(result, expected)

    def test_string_functions(self):
        string = ibis.literal('abcd')
        strip_string = ibis.literal('   a   ')

        cases = [
            (string.length(), 4),
            (ibis.literal('ABCD').lower(), 'abcd'),
            (string.upper(), 'ABCD'),
            (string.reverse(), 'dcba'),
            (string.ascii_str(), 97),
            (strip_string.strip(), 'a'),
            (strip_string.lstrip(), 'a   '),
            (strip_string.rstrip(), '   a'),
            (string.substr(0, 2), 'ab'),
            (string.left(2), 'ab'),
            (string.right(2), 'cd'),
            (string.repeat(2), 'abcdabcd'),
            (string.find('a'), 0),
            (ibis.literal('0123').translate('012', 'abc'), 'abc3'),
            (string.locate('a'), 0),
            (string.lpad(1, '-'), 'a'),
            (string.lpad(5, '-'), '-abcd'),
            (string.rpad(1, '-'), 'a'),
            (string.rpad(5, '-'), 'abcd-'),
            (string.find_in_set(['a', 'b', 'abcd']), 2),
            (ibis.literal(', ').join(['a', 'b']), 'a, b'),
            (string.like('a%'), True),
            (string.re_search('[a-z]'), True),
            (string.re_extract('[a-z]', 0), 'a'),
            (string.re_replace('(b)', '2'), 'a2cd'),
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected

    def test_filter_predicates(self):
        t = self.con.table('tpch_nation')

        predicates = [
            lambda x: x.n_name.lower().like('%ge%'),
            lambda x: x.n_name.lower().contains('ge'),
            lambda x: x.n_name.lower().rlike('.*ge.*')
        ]

        expr = t
        for pred in predicates:
            expr = expr[pred(expr)].projection([expr])

        expr.execute()

    def test_histogram_value_counts(self):
        t = self.alltypes
        expr = t.double_col.histogram(10).value_counts()
        expr.execute()

    def test_decimal_timestamp_builtins(self):
        table = self.con.table('tpch_lineitem')

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

            ts < (ibis.now() + ibis.month(3)),
            ts < (ibis.timestamp('2005-01-01') + ibis.month(3)),
        ]

        timestamp_fields = ['year', 'month', 'day', 'hour', 'minute',
                            'second', 'millisecond', 'microsecond',
                            'week']
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

    def test_timestamp_scalar_in_filter(self):
        # #310
        table = self.alltypes

        expr = (table.filter([table.timestamp_col <
                             (ibis.timestamp('2010-01-01') + ibis.month(3)),
                             table.timestamp_col < (ibis.now() + ibis.day(10))
                              ])
                .count())
        expr.execute()

    def test_aggregations_e2e(self):
        table = self.alltypes.limit(100)

        d = table.double_col
        s = table.string_col

        cond = table.string_col.isin(['1', '7'])

        exprs = [
            table.bool_col.count(),
            d.sum(),
            d.mean(),
            d.min(),
            d.max(),
            s.approx_nunique(),
            d.approx_median(),
            s.group_concat(),

            table.bool_col.count(where=cond),
            d.sum(where=cond),
            d.mean(where=cond),
            d.min(where=cond),
            d.max(where=cond),
        ]

        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()

    def test_tpch_self_join_failure(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

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
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

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

    def test_verbose_log_queries(self):
        queries = []

        def logger(x):
            queries.append(x)

        with config.option_context('verbose', True):
            with config.option_context('verbose_log', logger):
                self.con.table('tpch_orders', database=self.test_data_db)

        assert len(queries) == 1
        assert queries[0] == 'SELECT * FROM ibis_testing.`tpch_orders` LIMIT 0'


def _ensure_drop(con, table_name, database=None):
    con.drop_table(table_name, database=database, force=True)
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
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')
        table = self.con.parquet_file(hdfs_path)
        name = table.op().name
        table = None
        gc.collect()
        _assert_table_not_exists(self.con, name)

    def test_persist_parquet_file_with_name(self):
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

        name = _random_table_name()
        schema = ibis.schema([('r_regionkey', 'int16'),
                              ('r_name', 'string'),
                              ('r_comment', 'string')])
        self.con.parquet_file(hdfs_path, schema=schema,
                              name=name,
                              database=self.tmp_db,
                              persist=True)
        gc.collect()

        # table still exists
        self.con.table(name, database=self.tmp_db)

        _ensure_drop(self.con, name, database=self.tmp_db)

    def test_query_avro(self):
        hdfs_path = pjoin(self.test_data_dir, 'avro/tpch.region')

        avro_schema = {
            "fields": [
                {"type": ["int", "null"], "name": "R_REGIONKEY"},
                {"type": ["string", "null"], "name": "R_NAME"},
                {"type": ["string", "null"], "name": "R_COMMENT"}],
            "type": "record",
            "name": "a"
        }

        table = self.con.avro_file(hdfs_path, avro_schema,
                                   database=self.tmp_db)

        name = table.op().name
        assert 'ibis_tmp_' in name
        assert name.startswith('{0}.'.format(self.tmp_db))

        # table exists
        self.con.table(name)

        expr = table.r_name.value_counts()
        expr.execute()

        assert table.count().execute() == 5

        df = table.execute()
        assert len(df) == 5

    def test_query_parquet_file_with_schema(self):
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

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
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

        ex_schema = ibis.schema([('r_regionkey', 'int16'),
                                 ('r_name', 'string'),
                                 ('r_comment', 'string')])

        table = self.con.parquet_file(hdfs_path, like_table='tpch_region')

        assert table.schema().equals(ex_schema)

    def test_query_parquet_infer_schema(self):
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

        table = self.con.parquet_file(hdfs_path)

        # NOTE: the actual schema should have an int16, but bc this is being
        # inferred from a parquet file, which has no notion of int16, the
        # inferred schema will have an int32 instead.
        ex_schema = ibis.schema([('r_regionkey', 'int32'),
                                 ('r_name', 'string'),
                                 ('r_comment', 'string')])

        assert table.schema().equals(ex_schema)

    def test_query_text_file_regex(self):
        pass

    def test_query_delimited_file_directory(self):
        hdfs_path = pjoin(self.test_data_dir, 'csv')

        schema = ibis.schema([('foo', 'string'),
                              ('bar', 'double'),
                              ('baz', 'int8')])
        name = 'delimited_table_test1'
        table = self.con.delimited_file(hdfs_path, schema, name=name,
                                        database=self.tmp_db,
                                        delimiter=',')
        try:
            expr = (table
                    [table.bar > 0]
                    .group_by('foo')
                    .aggregate([table.bar.sum().name('sum(bar)'),
                                table.baz.sum().name('mean(baz)')]))
            expr.execute()
        finally:
            self.con.drop_table(name, database=self.tmp_db)
