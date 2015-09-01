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

from .common import SQLiteTests
from ibis.compat import unittest
from ibis import literal as L
import ibis

import sqlalchemy as sa


class TestSQLiteFunctions(SQLiteTests, unittest.TestCase):

    def test_cast(self):
        at = self._to_sqla(self.alltypes)

        d = self.alltypes.double_col
        s = self.alltypes.string_col

        sa_d = at.c.double_col
        sa_s = at.c.string_col

        cases = [
            (d.cast('int8'), sa.cast(sa_d, sa.types.SMALLINT)),
            (s.cast('double'), sa.cast(sa_s, sa.types.REAL)),
        ]
        self._check_expr_cases(cases)

    def test_decimal_cast(self):
        pass

    def test_binary_arithmetic(self):
        cases = [
            (L(3) + L(4), 7),
            (L(3) - L(4), -1),
            (L(3) * L(4), 12),
            (L(12) / L(4), 3),
            # (L(12) ** L(2), 144),
            (L(12) % L(5), 2)
        ]
        self._check_e2e_cases(cases)

    def test_aggregations_execute(self):
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

            table.bool_col.count(where=cond),
            d.sum(where=cond),
            d.mean(where=cond),
            d.min(where=cond),
            d.max(where=cond),

            s.group_concat(),
        ]

        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()
