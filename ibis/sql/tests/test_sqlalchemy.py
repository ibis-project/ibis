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
from ibis.tests.util import assert_equal
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.sql.alchemy as alch
import ibis.util as util
import ibis

from sqlalchemy import types as sat
import sqlalchemy.sql as sql
import sqlalchemy as sa

# SQL engine-independent unit tests


class MockAlchemyConnection(MockConnection):

    def __init__(self):
        self.meta = sa.MetaData()
        MockConnection.__init__(self)

    def table(self, name):
        schema = self._get_table_schema(name)
        table = alch.table_from_schema(name, self.meta, schema)
        node = alch.AlchemyTable(table, self)
        return ir.TableExpr(node)


class TestSQLAlchemy(unittest.TestCase):

    def setUp(self):
        self.con = MockAlchemyConnection()
        self.alltypes = self.con.table('functional_alltypes')
        self.sa_alltypes = self.con.meta.tables['functional_alltypes']
        self.meta = sa.MetaData()

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
            case = (f(self.alltypes.double_col, 5),
                    f(sat.c.double_col, 5))
            cases.append(case)

        self._check_expr_cases(cases)

    def test_boolean_conjunction(self):
        sat = self.sa_alltypes
        sd = sat.c.double_col

        d = self.alltypes.double_col
        cases = [
            ((d > 0) & (d < 5), sql.and_(sd > 0, sd < 5)),
            ((d < 0) | (d > 5), sql.or_(sd < 0, sd > 5))
        ]

        self._check_expr_cases(cases)

    def test_named_expr(self):
        sat = self.sa_alltypes
        d = self.alltypes.double_col

        cases = [
            ((d * 2).name('foo'), (sat.c.double_col * 2).label('foo'))
        ]
        self._check_expr_cases(cases, named=True)

    def test_joins(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        rt = self._to_sqla(region)
        nt = self._to_sqla(nation)

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

    def _compare_sqla(self, expr, sqla):
        result = alch.to_sqlalchemy(expr)
        assert str(result) == str(sqla)

    def _to_sqla(self, table):
        return table.op().sqla_table

    def test_outer_join(self):
        pass

    def test_left_right_join(self):
        pass

    def test_cte_extract(self):
        pass

    def test_self_reference(self):
        pass

    def test_uncorrelated_subquery(self):
        pass

    def test_general_sql_function(self):
        pass

    def test_union(self):
        pass
