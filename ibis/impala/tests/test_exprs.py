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

import pytest

import pandas as pd
from decimal import Decimal

import ibis

from ibis.compat import unittest, StringIO
from ibis.expr.datatypes import Category
from ibis.sql.compiler import to_sql
from ibis.impala.tests.common import ImpalaE2E

import ibis.expr.api as api


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


class TestImpalaExprs(ImpalaE2E, unittest.TestCase):

    def test_embedded_identifier_quoting(self):
        t = self.con.table('functional_alltypes')

        expr = (t[[(t.double_col * 2).name('double(fun)')]]
                ['double(fun)'].sum())
        expr.execute()

    def test_table_info(self):
        t = self.con.table('functional_alltypes')
        buf = StringIO()
        t.info(buf=buf)

        assert buf.getvalue() is not None

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
            if isinstance(x, Category):
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

    def test_aggregations(self):
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

            d.std(),
            d.std(how='pop'),
            d.var(),
            d.var(how='pop'),

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
            d.std(where=cond),
            d.var(where=cond),
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
