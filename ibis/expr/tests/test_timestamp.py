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

import pandas as pd

import ibis
import ibis.expr.api as api
import ibis.expr.operations as ops
import ibis.expr.types as ir

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest


class TestTimestamp(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.alltypes = self.con.table('alltypes')
        self.col = self.alltypes.i

    def test_field_select(self):
        assert isinstance(self.col, ir.TimestampArray)

    def test_string_cast_to_timestamp(self):
        casted = self.alltypes.g.cast('timestamp')
        assert isinstance(casted, ir.TimestampArray)

        string = api.literal('2000-01-01')
        casted = string.cast('timestamp')
        assert isinstance(casted, ir.TimestampScalar)

    def test_extract_fields(self):
        # type-size may be database specific
        cases = [
            ('year', ops.ExtractYear, ir.Int32Array),
            ('month', ops.ExtractMonth, ir.Int32Array),
            ('day', ops.ExtractDay, ir.Int32Array),
            ('hour', ops.ExtractHour, ir.Int32Array),
            ('minute', ops.ExtractMinute, ir.Int32Array),
            ('second', ops.ExtractSecond, ir.Int32Array),
            ('millisecond', ops.ExtractMillisecond, ir.Int32Array),
        ]

        for attr, ex_op, ex_type in cases:
            result = getattr(self.col, attr)()
            assert result.get_name() == attr
            assert isinstance(result, ex_type)
            assert isinstance(result.op(), ex_op)

    def test_now(self):
        result = api.now()
        assert isinstance(result, ir.TimestampScalar)
        assert isinstance(result.op(), ops.TimestampNow)

    def test_timestamp_literals(self):
        ts_str = '2015-01-01 00:00:00'
        val = pd.Timestamp(ts_str)

        expr = ibis.literal(val)
        assert isinstance(expr, ir.TimestampScalar)

        expr = ibis.timestamp(ts_str)
        assert isinstance(expr, ir.TimestampScalar)

        self.assertRaises(ValueError, ibis.timestamp, '2015-01-01 00:71')

    def test_integer_to_timestamp(self):
        # #246
        pass

    def test_comparison_timestamp(self):
        expr = self.col > (self.col.min() + ibis.day(3))
        assert isinstance(expr, ir.BooleanArray)

    def test_comparisons_string(self):
        val = '2015-01-01 00:00:00'
        expr = self.col > val
        op = expr.op()
        assert isinstance(op.right, ir.TimestampScalar)

        expr2 = val < self.col
        op = expr2.op()
        assert isinstance(op, ops.Greater)
        assert isinstance(op.right, ir.TimestampScalar)

    def test_comparisons_pandas_timestamp(self):
        val = pd.Timestamp('2015-01-01 00:00:00')
        expr = self.col > val
        op = expr.op()
        assert isinstance(op.right, ir.TimestampScalar)

        # TODO: this is broken for now because of upstream pandas problems

        # expr2 = val < self.col
        # op = expr2.op()
        # assert isinstance(op, ops.Greater)
        # assert isinstance(op.right, ir.TimestampScalar)
