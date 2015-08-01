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

import ibis

from ibis.compat import unittest
from ibis.sql.compiler import to_sql
from ibis.tests.util import IbisTestEnv, ImpalaE2E, assert_equal, connect_test

import ibis.common as com
import ibis.config as config
import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.util as util
import ibis.sql.udf as udf

from impala.error import HiveServer2Error as HS2Error


ENV = IbisTestEnv()


class TestImpalaConnection(ImpalaE2E, unittest.TestCase):

    def test_raise_ibis_error_no_hdfs(self):
        # #299
        client = connect_test(ENV, with_hdfs=False)
        self.assertRaises(com.IbisError, getattr, client, 'hdfs')

    def test_get_table_ref(self):
        table = self.db.functional_alltypes
        assert isinstance(table, ir.TableExpr)

        table = self.db['functional_alltypes']
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
        con = connect_test(env)
        self.assertRaises(Exception, con.table, 'functional_alltypes')
        con.set_database(self.test_data_db)
        con.table('functional_alltypes')

    def test_tables_robust_to_set_database(self):
        db_name = '__ibis_test_{0}'.format(util.guid())

        self.con.create_database(db_name)
        self.temp_databases.append(db_name)

        table = self.con.table('functional_alltypes')

        self.con.set_database(db_name)

        # it still works!
        table.limit(10).execute()

    def test_create_exists_drop_database(self):
        tmp_name = '__ibis_test_{0}'.format(util.guid())

        assert not self.con.exists_database(tmp_name)

        self.con.create_database(tmp_name)
        assert self.con.exists_database(tmp_name)

        self.con.drop_database(tmp_name)
        assert not self.con.exists_database(tmp_name)

    def test_exists_table(self):
        assert self.con.exists_table('functional_alltypes')
        assert not self.con.exists_table(util.guid())

    def test_create_exists_drop_view(self):
        tmp_name = util.guid()

        assert not self.con.exists_table(tmp_name)

        expr = (self.con.table('functional_alltypes')
                .group_by('string_col')
                .size())

        self.con.create_view(tmp_name, expr)
        self.temp_views.append(tmp_name)
        assert self.con.exists_table(tmp_name)

        # just check it works for now
        expr2 = self.con.table(tmp_name)
        expr2.execute()

        self.con.drop_view(tmp_name)
        assert not self.con.exists_table(tmp_name)

    def test_drop_non_empty_database(self):
        tmp_db = '__ibis_test_{0}'.format(util.guid())

        self.con.create_database(tmp_db)

        self.con.create_table(util.guid(), self.alltypes, database=tmp_db)

        # Has a view, too
        self.con.create_view(util.guid(), self.alltypes,
                             database=tmp_db)

        self.assertRaises(com.IntegrityError, self.con.drop_database, tmp_db)

        self.con.drop_database(tmp_db, force=True)
        assert not self.con.exists_database(tmp_db)

    def test_create_database_with_location(self):
        base = pjoin(self.tmp_dir, util.guid())
        name = '__ibis_test_{0}'.format(util.guid())
        tmp_path = pjoin(base, name)

        self.con.create_database(name, path=tmp_path)
        assert self.hdfs.exists(base)
        self.con.drop_database(name)
        self.hdfs.rmdir(base)

    def test_create_table_with_location(self):
        base = pjoin(self.tmp_dir, util.guid())
        name = 'test_{0}'.format(util.guid())
        tmp_path = pjoin(base, name)

        expr = self.alltypes
        table_name = _random_table_name()

        self.con.create_table(table_name, expr=expr, path=tmp_path,
                              database=self.test_data_db)
        self.temp_tables.append('.'.join([self.test_data_db, table_name]))
        assert self.hdfs.exists(tmp_path)

    def test_drop_table_not_exist(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_table, random_name)

        self.con.drop_table(random_name, force=True)

    def test_truncate_table(self):
        expr = self.alltypes.limit(50)

        table_name = util.guid()
        self.con.create_table(table_name, expr=expr)
        self.temp_tables.append(table_name)

        try:
            self.con.truncate_table(table_name)
        except HS2Error as e:
            if 'AnalysisException' in e.message:
                pytest.skip('TRUNCATE not available in this '
                            'version of Impala')

        result = self.con.table(table_name).execute()
        assert len(result) == 0

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

    def test_create_empty_table(self):
        schema = ibis.schema([('a', 'string'),
                              ('b', 'timestamp'),
                              ('c', 'decimal(12,8)'),
                              ('d', 'double')])

        table_name = util.guid()
        self.con.create_table(table_name, schema=schema)
        self.temp_tables.append(table_name)

        result_schema = self.con.get_schema(table_name)
        assert_equal(result_schema, schema)

        assert len(self.con.table(table_name).execute()) == 0

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

            # hash functions
            i4.hash(),
            d.hash(),
            s.hash(),

            # modulus cases
            i1 % 5,
            i4 % 10,
            20 % i1,
            d % 5,

            i1.zeroifnull(),
            i4.zeroifnull(),
            i8.zeroifnull(),

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

            i1.sign(),
            i4.sign(),
            d.sign(),

            # conv
            i1.convert_base(10, 2),
            i4.convert_base(10, 2),
            i8.convert_base(10, 2),
            s.convert_base(10, 2),

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
            s.translate("a", "b"),
            s.find("a"),
            s.lpad(10, 'a'),
            s.rpad(10, 'a'),
            s.find_in_set(["a"]),
            s.lower(),
            s.upper(),
            s.reverse(),
            s.ascii_str(),
            s.length(),
            s.strip(),
            s.lstrip(),
            s.strip(),

            # strings with int expr inputs
            s.left(i1),
            s.right(i1),
            s.substr(i1, i1 + 2),
            s.repeat(i1)
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs]
        projection.limit(10).execute()

        self._check_impala_output_types_match(projection)

    def _check_impala_output_types_match(self, table):
        query = to_sql(table)
        t = self.con.sql(query)

        def _clean_type(x):
            if isinstance(x, ir.CategoryType):
                x = x.to_integer_type()
            return x

        left, right = t.schema(), table.schema()
        for i, (n, l, r) in enumerate(zip(left.names, left.types,
                                          right.types)):
            l = _clean_type(l)
            r = _clean_type(r)

            if l != r:
                pytest.fail('Value for {0} had left type {1}'
                            ' and right type {2}'.format(n, l, r))

    def assert_cases_equality(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected, to_sql(expr)

    def test_int_builtins(self):
        i8 = ibis.literal(50)
        i32 = ibis.literal(50000)

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

        self.assert_cases_equality(mod_cases + nullif_cases)

    def test_column_types(self):
        df = self.alltypes.execute()
        assert df.tinyint_col.dtype.name == 'int8'
        assert df.smallint_col.dtype.name == 'int16'
        assert df.int_col.dtype.name == 'int32'
        assert df.bigint_col.dtype.name == 'int64'
        assert df.float_col.dtype.name == 'float32'
        assert df.double_col.dtype.name == 'float64'
        assert pd.core.common.is_datetime64_dtype(df.timestamp_col.dtype)

    def test_timestamp_builtins(self):
        i32 = ibis.literal(50000)
        i64 = ibis.literal(5 * 10 ** 8)

        stamp = ibis.timestamp('2009-05-17 12:34:56')

        timestamp_cases = [
            (i32.to_timestamp('s'), pd.to_datetime(50000, unit='s')),
            (i32.to_timestamp('ms'), pd.to_datetime(50000, unit='ms')),
            (i64.to_timestamp(), pd.to_datetime(5 * 10 ** 8, unit='s')),

            (stamp.truncate('y'), pd.Timestamp('2009-01-01')),
            (stamp.truncate('m'), pd.Timestamp('2009-05-01')),
            (stamp.truncate('d'), pd.Timestamp('2009-05-17')),
            (stamp.truncate('h'), pd.Timestamp('2009-05-17 12:00')),
            (stamp.truncate('minute'), pd.Timestamp('2009-05-17 12:34'))
        ]

        self.assert_cases_equality(timestamp_cases)

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
            (d.round(2), Decimal('5.25')),
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
            (dc.zeroifnull(), Decimal('5.245')),
            (-dc, Decimal('-5.245'))
        ]

        def approx_equal(a, b, eps):
            assert abs(a - b) < eps

        for expr, expected in cases:
            result = self.con.execute(expr)
            if isinstance(expected, Decimal):
                tol = Decimal('0.0001')
            else:
                tol = 0.0001
            approx_equal(result, expected, tol)

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
            (string.capitalize(), 'Abcd'),
            (string.substr(0, 2), 'ab'),
            (string.left(2), 'ab'),
            (string.right(2), 'cd'),
            (string.repeat(2), 'abcdabcd'),
            (ibis.literal('0123').translate('012', 'abc'), 'abc3'),
            (string.find('a'), 0),
            (ibis.literal('baaaab').find('b', 2), 5),
            (string.lpad(1, '-'), 'a'),
            (string.lpad(5), ' abcd'),
            (string.rpad(1, '-'), 'a'),
            (string.rpad(5), 'abcd '),
            (string.find_in_set(['a', 'b', 'abcd']), 2),
            (ibis.literal(', ').join(['a', 'b']), 'a, b'),
            (string.like('a%'), True),
            (string.re_search('[a-z]'), True),
            (ibis.literal("https://www.cloudera.com").parse_url('HOST'),
             "www.cloudera.com"),
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

            # hashing
            dc.hash(),
            ts.hash(),

            # truncate
            ts.truncate('y'),
            ts.truncate('q'),
            ts.truncate('month'),
            ts.truncate('d'),
            ts.truncate('w'),
            ts.truncate('h'),
            ts.truncate('minute'),
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

            table.bool_col.any(),
            table.bool_col.notany(),
            -table.bool_col.any(),

            table.bool_col.all(),
            table.bool_col.notall(),
            -table.bool_col.all(),

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

    def test_analytic_functions(self):
        t = self.alltypes.limit(1000)

        g = t.group_by('string_col').order_by('double_col')
        f = t.float_col

        exprs = [
            f.lag(),
            f.lead(),
            f.rank(),
            f.dense_rank(),

            f.first(),
            f.last(),

            f.first().over(ibis.window(preceding=10)),
            f.first().over(ibis.window(following=10)),

            ibis.row_number(),
            f.cumsum(),
            f.cummean(),
            f.cummin(),
            f.cummax(),

            # boolean cumulative reductions
            (f == 0).cumany(),
            (f == 0).cumall(),

            f.sum(),
            f.mean(),
            f.min(),
            f.max()
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        proj_table = g.mutate(proj_exprs)
        proj_table.execute()

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

        # no analysis failure
        self.con.explain(yoy)

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
        self.con.explain(expr)

    def test_non_equijoin(self):
        t = self.con.table('functional_alltypes').limit(100)
        t2 = t.view()

        expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

        # it works
        expr.execute()

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


class TestPartitioning(ImpalaE2E, unittest.TestCase):

    def test_create_table_with_partition_column(self):
        schema = ibis.schema([('year', 'int32'),
                              ('month', 'int8'),
                              ('day', 'int8'),
                              ('value', 'double')])

        name = util.guid()
        self.con.create_table(name, schema=schema, partition=['year', 'month'])
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'int8')])
        table_schema = self.con.get_schema(name)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.con.get_partition_schema(name)
        expected = ibis.schema([('year', 'int32'),
                                ('month', 'int8')])
        assert_equal(partition_schema, expected)



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
        hdfs_path = pjoin(self.test_data_dir, 'avro/tpch_region_avro')

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

        assert_equal(table.schema(), ex_schema)

    def test_query_parquet_infer_schema(self):
        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')
        table = self.con.parquet_file(hdfs_path)

        # NOTE: the actual schema should have an int16, but bc this is being
        # inferred from a parquet file, which has no notion of int16, the
        # inferred schema will have an int32 instead.
        ex_schema = ibis.schema([('r_regionkey', 'int32'),
                                 ('r_name', 'string'),
                                 ('r_comment', 'string')])

        assert_equal(table.schema(), ex_schema)

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

    def test_temp_table_concurrency(self):
        pytest.skip('Cannot get this test to run under pytest')

        from threading import Thread, Lock
        import gc
        nthreads = 4

        hdfs_path = pjoin(self.test_data_dir, 'parquet/tpch_region')

        lock = Lock()

        results = []

        def do_something():
            t = self.con.parquet_file(hdfs_path)

            with lock:
                t.limit(10).execute()
                t = None
                gc.collect()
                results.append(True)

        threads = []
        for i in range(nthreads):
            t = Thread(target=do_something)
            t.start()
            threads.append(t)

        [x.join() for x in threads]

        assert results == [True] * nthreads


class TestObjectLayer(ImpalaE2E, unittest.TestCase):

    def test_database_repr(self):
        assert self.test_data_db in repr(self.db)

    def test_database_drop(self):
        tmp_name = '__ibis_test_{0}'.format(util.guid())
        self.con.create_database(tmp_name)

        db = self.con.database(tmp_name)
        self.temp_databases.append(tmp_name)
        db.drop()
        assert not self.con.exists_database(tmp_name)

    def test_compute_stats(self):
        self.con.table('functional_alltypes').compute_stats()

    def test_namespace(self):
        ns = self.db.namespace('tpch_')

        assert 'tpch_' in repr(ns)

        table = ns.lineitem
        expected = self.db.tpch_lineitem
        attrs = dir(ns)
        assert 'lineitem' in attrs
        assert 'functional_alltypes' not in attrs

        assert_equal(table, expected)

    def test_drop_table_or_view(self):
        t = self.db.functional_alltypes

        tname = util.guid()
        self.con.create_table(tname, t.limit(10))
        self.temp_tables.append(tname)

        vname = util.guid()
        self.con.create_view(vname, t.limit(10))
        self.temp_views.append(vname)

        t2 = self.db[tname]
        t2.drop()
        assert tname not in self.db

        t3 = self.db[vname]
        t3.drop()
        assert vname not in self.db

    def test_udf(self):
        pass

    def test_uda(self):
        pass


class TestUDFWrapping(ImpalaE2E, unittest.TestCase):

    def setUp(self):
        super(TestUDFWrapping, self).setUp()
        self.udf_so = self.test_data_dir + '/udf/libudfsample.so'
        self.uda_so = self.test_data_dir + '/udf/libudasample.so'

    def test_boolean_wrapping(self):
        col = self.alltypes.bool_col
        literal = ibis.literal(True)
        self._identity_func_testing('boolean', literal, col)

    def test_tinyint_wrapping(self):
        col = self.alltypes.tinyint_col
        literal = ibis.literal(5)
        self._identity_func_testing('int8', literal, col)

    def test_int_wrapping(self):
        col = self.alltypes.int_col
        literal = ibis.literal(1000)
        self._identity_func_testing('int32', literal, col)

    def test_bigint_wrapping(self):
        col = self.alltypes.bigint_col
        literal = ibis.literal(1000).cast('int64')
        self._identity_func_testing('int64', literal, col)

    def test_float_wrapping(self):
        col = self.alltypes.float_col
        literal = ibis.literal(3.14)
        self._identity_func_testing('float', literal, col)

    def test_double_wrapping(self):
        col = self.alltypes.double_col
        literal = ibis.literal(3.14)
        self._identity_func_testing('double', literal, col)

    def test_string_wrapping(self):
        col = self.alltypes.string_col
        literal = ibis.literal('ibis')
        self._identity_func_testing('string', literal, col)

    def test_timestamp_wrapping(self):
        col = self.alltypes.timestamp_col
        literal = ibis.timestamp('1961-04-10')
        self._identity_func_testing('timestamp', literal, col)

    def test_decimal_wrapping(self):
        col = self.con.table('tpch_customer').c_acctbal
        literal = ibis.literal(1).cast('decimal(12,2)')
        op = self._udf_creation_to_op('identity', 'Identity',
                                      ['decimal(12,2)'], 'decimal(12,2)')

        def _func(val):
            return op(val).to_expr()
        expr = _func(literal)
        assert issubclass(type(expr), ir.ScalarExpr)
        result = self.con.execute(expr)
        assert result == Decimal(1)

        expr = _func(col)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

    def test_mixed_inputs(self):
        name = 'two_args'
        symbol = 'TwoArgs'
        inputs = ['int32', 'int32']
        output = 'int32'
        op = self._udf_creation_to_op(name, symbol, inputs, output)

        def _two_args(val1, val2):
            return op(val1, val2).to_expr()

        expr = _two_args(self.alltypes.int_col, 1)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

        expr = _two_args(1, self.alltypes.int_col)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

        expr = _two_args(self.alltypes.int_col, self.alltypes.tinyint_col)
        self.con.execute(expr)

    def test_implicit_typecasting(self):
        col = self.alltypes.tinyint_col
        literal = ibis.literal(1000)
        self._identity_func_testing('int32', literal, col)

    def test_mult_type_args_wrapping(self):
        symbol = 'AlmostAllTypes'
        name = 'most_types'
        inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
                  'int64', 'float', 'double']
        output = 'int32'

        op = self._udf_creation_to_op(name, symbol, inputs, output)

        def _mult_types(string, boolean, tinyint, smallint, integer,
                        bigint, float_val, double_val):
            return op(string, boolean, tinyint, smallint, integer,
                      bigint, float_val, double_val).to_expr()
        expr = _mult_types('a', True, 1, 1, 1, 1, 1.0, 1.0)
        result = self.con.execute(expr)
        assert result == 8

        table = self.alltypes
        expr = _mult_types(table.string_col, table.bool_col,
                           table.tinyint_col, table.tinyint_col,
                           table.smallint_col, table.smallint_col,
                           1.0, 1.0)
        self.con.execute(expr)

    def test_drop_udf_not_exists(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_udf, random_name)

    def _udf_creation_to_op(self, name, symbol, inputs, output):
        udf_info = udf.UDFCreator(self.udf_so, inputs, output, symbol, name)
        self.temp_functions.append((name, inputs))
        self.con.create_udf(udf_info, database=self.test_data_db)
        op = udf_info.to_operation()
        udf.add_impala_operation(op, name, self.test_data_db)
        assert self.con.exists_udf(name, self.test_data_db)
        return op

    def _identity_func_testing(self, datatype, literal, column):
        inputs = [datatype]
        name = 'identity'
        op = self._udf_creation_to_op(name, 'Identity', inputs, datatype)

        def _identity_test(value):
            return op(value).to_expr()
        expr = _identity_test(literal)
        assert issubclass(type(expr), ir.ScalarExpr)
        result = self.con.execute(expr)
        # Hacky
        if datatype is 'timestamp':
            assert type(result) == pd.tslib.Timestamp
        else:
            self.assertEqual(result, literal)

        expr = _identity_test(column)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)
