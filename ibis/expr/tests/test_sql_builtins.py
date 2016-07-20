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

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest
from ibis.tests.util import assert_equal
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis


class TestBuiltins(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.alltypes = self.con.table('functional_alltypes')
        self.lineitem = self.con.table('tpch_lineitem')

    def test_abs(self):
        colnames = ['tinyint_col', 'smallint_col', 'int_col', 'bigint_col',
                    'float_col', 'double_col']

        fname = 'abs'
        op = ops.Abs

        for col in colnames:
            expr = self.alltypes[col]
            self._check_unary_op(expr, fname, op, type(expr))

        expr = self.lineitem.l_extendedprice
        self._check_unary_op(expr, fname, op, type(expr))

    def test_group_concat(self):
        col = self.alltypes.string_col

        expr = col.group_concat()
        assert isinstance(expr.op(), ops.GroupConcat)
        arg, sep = expr.op().args
        sep == ','

        expr = col.group_concat('|')
        arg, sep = expr.op().args
        sep == '|'

    def test_zeroifnull(self):
        dresult = self.alltypes.double_col.zeroifnull()
        iresult = self.alltypes.int_col.zeroifnull()

        assert type(dresult.op()) == ops.ZeroIfNull
        assert type(dresult) == ir.DoubleArray

        # Impala upconverts all ints to bigint. Hmm.
        assert type(iresult) == type(iresult)

    def test_fillna(self):
        result = self.alltypes.double_col.fillna(5)
        assert isinstance(result, ir.DoubleArray)

        assert isinstance(result.op(), ops.IfNull)

        result = self.alltypes.bool_col.fillna(True)
        assert isinstance(result, ir.BooleanArray)

        # Retains type of caller (for now)
        result = self.alltypes.int_col.fillna(self.alltypes.bigint_col)
        assert isinstance(result, ir.Int32Array)

    def test_ceil_floor(self):
        cresult = self.alltypes.double_col.ceil()
        fresult = self.alltypes.double_col.floor()
        assert isinstance(cresult, ir.Int64Array)
        assert isinstance(fresult, ir.Int64Array)
        assert type(cresult.op()) == ops.Ceil
        assert type(fresult.op()) == ops.Floor

        cresult = ibis.literal(1.2345).ceil()
        fresult = ibis.literal(1.2345).floor()
        assert isinstance(cresult, ir.Int64Scalar)
        assert isinstance(fresult, ir.Int64Scalar)

        dec_col = self.lineitem.l_extendedprice
        cresult = dec_col.ceil()
        fresult = dec_col.floor()
        assert isinstance(cresult, ir.DecimalArray)
        assert cresult.meta == dec_col.meta

        assert isinstance(fresult, ir.DecimalArray)
        assert fresult.meta == dec_col.meta

    def test_sign(self):
        result = self.alltypes.double_col.sign()
        assert isinstance(result, ir.FloatArray)
        assert type(result.op()) == ops.Sign

        result = ibis.literal(1.2345).sign()
        assert isinstance(result, ir.FloatScalar)

        dec_col = self.lineitem.l_extendedprice
        result = dec_col.sign()
        assert isinstance(result, ir.FloatArray)

    def test_round(self):
        result = self.alltypes.double_col.round()
        assert isinstance(result, ir.Int64Array)
        assert result.op().args[1] is None

        result = self.alltypes.double_col.round(2)
        assert isinstance(result, ir.DoubleArray)
        assert result.op().args[1].equals(ibis.literal(2))

        # Even integers are double (at least in Impala, check with other DB
        # implementations)
        result = self.alltypes.int_col.round(2)
        assert isinstance(result, ir.DoubleArray)

        dec = self.lineitem.l_extendedprice
        result = dec.round()
        assert isinstance(result, ir.DecimalArray)

        result = dec.round(2)
        assert isinstance(result, ir.DecimalArray)

        result = ibis.literal(1.2345).round()
        assert isinstance(result, ir.Int64Scalar)

    def _check_unary_op(self, expr, fname, ex_op, ex_type):
        result = getattr(expr, fname)()
        assert type(result.op()) == ex_op
        assert type(result) == ex_type


class TestCoalesceLikeFunctions(unittest.TestCase):

    def setUp(self):
        self.table = ibis.table([
            ('v1', 'decimal(12, 2)'),
            ('v2', 'decimal(10, 4)'),
            ('v3', 'int32'),
            ('v4', 'int64'),
            ('v5', 'float'),
            ('v6', 'double'),
            ('v7', 'string'),
            ('v8', 'boolean')
        ], 'testing')

        self.functions = [ibis.coalesce, ibis.greatest, ibis.least]

    def test_coalesce_instance_method(self):
        v7 = self.table.v7
        v5 = self.table.v5.cast('string')
        v8 = self.table.v8.cast('string')

        result = v7.coalesce(v5, v8, 'foo')
        expected = ibis.coalesce(v7, v5, v8, 'foo')
        assert_equal(result, expected)

    def test_integer_promotions(self):
        t = self.table

        for f in self.functions:
            expr = f(t.v3, t.v4)
            assert isinstance(expr, ir.Int64Array)

            expr = f(5, t.v3)
            assert isinstance(expr, ir.Int64Array)

            expr = f(5, 12)
            assert isinstance(expr, ir.Int64Scalar)

    def test_floats(self):
        t = self.table

        for f in self.functions:
            expr = f(t.v5)
            assert isinstance(expr, ir.DoubleArray)

            expr = f(5.5, t.v5)
            assert isinstance(expr, ir.DoubleArray)

            expr = f(5.5, 5)
            assert isinstance(expr, ir.DoubleScalar)

    def test_bools(self):
        pass

    def test_decimal_promotions(self):
        pass
