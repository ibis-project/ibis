# Copyright 2015 Cloudera Inc.
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

import operator

from ibis.compat import unittest
from ibis.expr.tests.mocks import MockConnection
from ibis.sql.tests.test_compiler import ExprTestCases
from ibis.tests.util import assert_equal
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.sql.alchemy as alch
import ibis

from sqlalchemy import types as sat, func as F
import sqlalchemy.sql as sql
import sqlalchemy as sa

L = sa.literal

# SQL engine-independent unit tests


class MockAlchemyConnection(MockConnection):

    def __init__(self):
        self.meta = sa.MetaData()
        MockConnection.__init__(self)

    def table(self, name):
        schema = self._get_table_schema(name)
        return self._inject_table(name, schema)

    def _inject_table(self, name, schema):
        if name in self.meta.tables:
            table = self.meta.tables[name]
        else:
            table = alch.table_from_schema(name, self.meta, schema)

        node = alch.AlchemyTable(table, self)
        return ir.TableExpr(node)


def _table_wrapper(name, tname=None):
    @property
    def f(self):
        t = self._table_from_schema(name, tname)
        return t
    return f


class TestSQLAlchemySelect(unittest.TestCase, ExprTestCases):

    def setUp(self):
        self.con = MockAlchemyConnection()
        self.alltypes = self.con.table('functional_alltypes')
        self.sa_alltypes = self.con.meta.tables['functional_alltypes']
        self.meta = sa.MetaData()

        self.sa_star1 = self._get_sqla('star1')

    foo = _table_wrapper('foo')
    bar = _table_wrapper('bar')
    t1 = _table_wrapper('t1', 'foo')
    t2 = _table_wrapper('t2', 'bar')

    def _table_from_schema(self, name, tname=None):
        tname = tname or name
        schema = ibis.schema(self._schemas[name])
        return self.con._inject_table(tname, schema)

    def _get_sqla(self, name):
        return self._to_sqla(self.con.table(name))

    def _check_expr_cases(self, cases, named=False):
        for expr, expected in cases:
            result = self._translate(expr, named=named)
            assert str(result) == str(expected)
            if named:
                assert result.name == expected.name

    def _translate(self, expr, named=False, context=None):
        translator = alch.AlchemyExprTranslator(expr, context=context,
                                                named=named)
        return translator.get_result()

    def test_sqla_schema_conversion(self):
        typespec = [
            # name, type, nullable
            ('smallint', sat.SmallInteger, False, dt.int16),
            ('int', sat.Integer, True, dt.int32),
            ('integer', sat.INTEGER(), True, dt.int64),
            ('bigint', sat.BigInteger, False, dt.int64),
            ('real', sat.REAL, True, dt.double),
            ('bool', sat.Boolean, True, dt.boolean),
            ('timestamp', sat.DateTime, True, dt.timestamp),
        ]

        sqla_types = []
        ibis_types = []
        for name, t, nullable, ibis_type in typespec:
            sqla_type = sa.Column(name, t, nullable=nullable)
            sqla_types.append(sqla_type)
            ibis_types.append((name, ibis_type(nullable)))

        table = sa.Table('tname', self.meta, *sqla_types)

        schema = alch.schema_from_table(table)
        expected = ibis.schema(ibis_types)

        assert_equal(schema, expected)

    def test_ibis_to_sqla_conversion(self):
        pass

    def test_comparisons(self):
        sat = self.sa_alltypes

        ops = ['ge', 'gt', 'lt', 'le', 'eq', 'ne']

        cases = []

        for op in ops:
            f = getattr(operator, op)
            case = f(self.alltypes.double_col, 5), f(sat.c.double_col, L(5))
            cases.append(case)

        self._check_expr_cases(cases)

    def test_boolean_conjunction(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col

        d = self.alltypes.double_col
        cases = [
            ((d > 0) & (d < 5), sql.and_(sd > L(0), sd < L(5))),
            ((d < 0) | (d > 5), sql.or_(sd < L(0), sd > L(5)))
        ]

        self._check_expr_cases(cases)

    def test_between(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col
        d = self.alltypes.double_col

        cases = [
            (d.between(5, 10), sd.between(L(5), L(10))),
        ]
        self._check_expr_cases(cases)

    def test_isnull_notnull(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col
        d = self.alltypes.double_col

        cases = [
            (d.isnull(), sd.is_(sa.null())),
            (d.notnull(), sd.isnot(sa.null())),
        ]
        self._check_expr_cases(cases)

    def test_negate(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col
        d = self.alltypes.double_col
        cases = [
            (-(d > 0), sql.not_(sd > L(0)))
        ]

        self._check_expr_cases(cases)

    def test_coalesce(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col
        sf = sat.c.float_col

        d = self.alltypes.double_col
        f = self.alltypes.float_col
        null = sa.null()

        v1 = ibis.NA
        v2 = (d > 30).ifelse(d, ibis.NA)
        v3 = f

        cases = [
            (ibis.coalesce(v2, v1, v3),
             sa.func.coalesce(sa.case([(sd > L(30), sd)], else_=null),
                              null, sf))
        ]
        self._check_expr_cases(cases)

    def test_named_expr(self):
        sat = self.sa_alltypes
        d = self.alltypes.double_col

        cases = [
            ((d * 2).name('foo'), (sat.c.double_col * L(2)).label('foo'))
        ]
        self._check_expr_cases(cases, named=True)

    def test_joins(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        rt = self._to_sqla(region).alias('t0')
        nt = self._to_sqla(nation).alias('t1')

        ipred = region.r_regionkey == nation.n_regionkey
        spred = rt.c.r_regionkey == nt.c.n_regionkey

        joins = [
            (region.inner_join(nation, ipred),
             rt.join(nt, spred)),

            (region.left_join(nation, ipred),
             rt.join(nt, spred, isouter=True)),

            (region.outer_join(nation, ipred),
             rt.outerjoin(nt, spred)),
        ]
        for ibis_joined, joined_sqla in joins:
            expected = sa.select([rt, nt]).select_from(joined_sqla)
            self._compare_sqla(ibis_joined, expected)

    def test_simple_case(self):
        self.con.table('alltypes')
        st = self.con.meta.tables['alltypes']

        expr = self._case_simple_case()

        cases = [
            (expr, sa.case([(st.c.g == L('foo'), L('bar')),
                            (st.c.g == L('baz'), L('qux'))],
                           else_='default')),
        ]
        self._check_expr_cases(cases)

    def test_searched_case(self):
        self.con.table('alltypes')
        st = self.con.meta.tables['alltypes']

        expr = self._case_search_case()
        cases = [
            (expr, sa.case([(st.c.f > L(0), st.c.d * L(2)),
                            (st.c.c < L(0), st.c.a * L(2))],
                           else_=sa.null())),
        ]
        self._check_expr_cases(cases)

    def test_where_simple_comparisons(self):
        expr = self._case_where_simple_comparisons()

        st = self.sa_star1.alias('t0')

        clause = sql.and_(st.c.f > L(0), st.c.c < (st.c.f * L(2)))
        expected = sa.select([st]).where(clause)

        self._compare_sqla(expr, expected)

    def test_simple_aggregate_query(self):
        st = self.sa_star1.alias('t0')

        cases = self._case_simple_aggregate_query()

        metric = F.sum(st.c.f).label('total')
        k1 = st.c.foo_id
        k2 = st.c.bar_id
        expected = [
            sa.select([k1, metric]).group_by(k1),
            sa.select([k1, k2, metric]).group_by(k1, k2)
        ]

        for case, ex_sqla in zip(cases, expected):
            self._compare_sqla(case, ex_sqla)

    def test_aggregate_having(self):
        st = self.sa_star1.alias('t0')

        cases = self._case_aggregate_having()

        metric = F.sum(st.c.f)
        k1 = st.c.foo_id
        expected = [
            sa.select([k1, metric.label('total')]).group_by(k1)
            .having(metric > L(10)),
            sa.select([k1, metric.label('total')]).group_by(k1)
            .having(F.count('*') > L(100))
        ]

        for case, ex_sqla in zip(cases, expected):
            self._compare_sqla(case, ex_sqla)

    def test_sort_by(self):
        st = self.sa_star1.alias('t0')
        cases = self._case_sort_by()

        base = sa.select([st])
        expected = [
            base.order_by(st.c.f),
            base.order_by(st.c.f.desc()),
            base.order_by(st.c.c, st.c.f.desc()),
        ]
        for case, ex_sqla in zip(cases, expected):
            self._compare_sqla(case, ex_sqla)

    def test_limit(self):
        cases = self._case_limit()

        st = self.sa_star1.alias('t0')
        base = sa.select([st])

        expected = [
            base.limit(10),
            base.limit(10).offset(5),
            base.where(st.c.f > L(0)).limit(10),
        ]

        st = self.sa_star1.alias('t1')
        base = sa.select([st])
        aliased = base.limit(10).alias('t0')
        case4 = sa.select([aliased]).where(aliased.c.f > L(0))
        expected.append(case4)

        for case, ex in zip(cases, expected):
            self._compare_sqla(case, ex)

    def test_cte_factor_distinct_but_equal(self):
        expr = self._case_cte_factor_distinct_but_equal()

        alltypes = self._get_sqla('alltypes')

        t2 = alltypes.alias('t2')
        t0 = (sa.select([t2.c.g, F.sum(t2.c.f).label('metric')])
              .group_by(t2.c.g)
              .cte('t0'))

        t1 = t0.alias('t1')
        table_set = t0.join(t1, t0.c.g == t1.c.g)
        stmt = sa.select([t0]).select_from(table_set)

        self._compare_sqla(expr, stmt)

    def test_self_reference(self):
        pass

    def test_where_uncorrelated_subquery(self):
        expr = self._case_where_uncorrelated_subquery()

        foo = self._to_sqla(self.foo).alias('t0')
        bar = self._to_sqla(self.bar)

        subq = sa.select([bar.c.job])
        stmt = sa.select([foo]).where(foo.c.job.in_(subq))
        self._compare_sqla(expr, stmt)

    def test_where_correlated_subquery(self):
        expr = self._case_where_correlated_subquery()

        foo = self._to_sqla(self.foo)
        t0 = foo.alias('t0')
        t1 = foo.alias('t1')
        subq = (sa.select([F.avg(t1.c.y).label('mean')])
                .where(t0.c.dept_id == t1.c.dept_id))
        stmt = sa.select([t0]).where(t0.c.y > subq)
        self._compare_sqla(expr, stmt)

    def test_exists(self):
        e1, e2 = self._case_exists()

        t1 = self._to_sqla(self.t1).alias('t0')
        t2 = self._to_sqla(self.t2).alias('t1')

        cond1 = sa.exists([L(1)]).where(t1.c.key1 == t2.c.key1)
        ex1 = sa.select([t1]).where(cond1)

        cond2 = sa.exists([L(1)]).where(
            sql.and_(t1.c.key1 == t2.c.key1, t2.c.key2 == L('foo')))
        ex2 = sa.select([t1]).where(cond2)

        # pytest.skip('not yet implemented')

        self._compare_sqla(e1, ex1)
        self._compare_sqla(e2, ex2)

    def test_not_exists(self):
        expr = self._case_not_exists()

        t1 = self._to_sqla(self.t1).alias('t0')
        t2 = self._to_sqla(self.t2).alias('t1')

        cond1 = sa.exists([L(1)]).where(t1.c.key1 == t2.c.key1)
        expected = sa.select([t1]).where(-cond1)

        self._compare_sqla(expr, expected)

    def test_general_sql_function(self):
        pass

    def test_union(self):
        pass

    def test_table_distinct(self):
        t = self.alltypes
        sat = self.sa_alltypes.alias('t0')

        cases = [
            (t.distinct(), sa.select([sat]).distinct()),
            (t['string_col', 'int_col'].distinct(),
             sa.select([sat.c.string_col, sat.c.int_col]).distinct())
        ]
        for case, ex in cases:
            self._compare_sqla(case, ex)

    def test_array_distinct(self):
        t = self.alltypes
        sat = self.sa_alltypes.alias('t0')

        cases = [
            (t.string_col.distinct(),
             sa.select([sat.c.string_col.distinct()]))
        ]
        for case, ex in cases:
            self._compare_sqla(case, ex)

    def test_count_distinct(self):
        t = self.alltypes
        sat = self.sa_alltypes.alias('t0')

        cases = [
            (t.int_col.nunique().name('nunique'),
             sa.select([F.count(sat.c.int_col.distinct())
                        .label('nunique')])),
            (t.group_by('string_col')
             .aggregate(t.int_col.nunique().name('nunique')),
             sa.select([sat.c.string_col,
                        F.count(sat.c.int_col.distinct())
                        .label('nunique')])
             .group_by(sat.c.string_col)),
        ]
        for case, ex in cases:
            self._compare_sqla(case, ex)

    def _compare_sqla(self, expr, sqla):
        result = alch.to_sqlalchemy(expr)
        assert str(result.compile()) == str(sqla.compile())

    def _to_sqla(self, table):
        return table.op().sqla_table
