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

import unittest

import ibis.expr.base as api
import ibis.expr.base as ir

from ibis.expr.tests.mocks import MockConnection


class TestDecimal(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.lineitem = self.con.table('tpch_lineitem')

    def test_type_metadata(self):
        col = self.lineitem.l_extendedprice
        assert isinstance(col, ir.DecimalArray)

        assert col.precision == 12
        assert col.scale == 2

    def test_cast_scalar_to_decimal(self):
        val = api.literal('1.2345')

        casted = val.cast('decimal(15,5)')
        assert isinstance(casted, ir.DecimalScalar)
        assert casted.precision == 15
        assert casted.scale == 5

    def test_decimal_aggregate_function_behavior(self):
        # From the Impala documentation: "The result of an aggregate function
        # such as MAX(), SUM(), or AVG() on DECIMAL values is promoted to a
        # scale of 38, with the same precision as the underlying column. Thus,
        # the result can represent the largest possible value at that
        # particular precision."
        col = self.lineitem.l_extendedprice
        functions = ['sum', 'mean', 'max', 'min']

        for func_name in functions:
            result = getattr(col, func_name)()
            assert isinstance(result, ir.DecimalScalar)
            assert result.precision == col.precision
            assert result.scale == 38

    def test_invalid_precision_scale_combo(self):
        pass
