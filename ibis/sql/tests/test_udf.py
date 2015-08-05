# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ibis

import ibis.sql.udf as udf
import ibis.expr.types as ir

from ibis.compat import unittest
from ibis.expr.tests.mocks import MockConnection
from ibis.sql.exprs import _operation_registry
from ibis.expr.operations import ValueOp
from ibis.common import IbisTypeError


class UDFTest(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

        self.i8 = self.table.tinyint_col
        self.i16 = self.table.smallint_col
        self.i32 = self.table.int_col
        self.i64 = self.table.bigint_col
        self.d = self.table.double_col
        self.f = self.table.float_col
        self.s = self.table.string_col
        self.b = self.table.bool_col
        self.t = self.table.timestamp_col
        self.dec = self.con.table('tpch_customer').c_acctbal
        self.all_cols = [self.i8, self.i16, self.i32, self.i64, self.d,
                         self.f, self.dec, self.s, self.b, self.t]

    def test_sql_generation(self):
        op = udf.scalar_function(['string'], 'string', name='Tester')
        udf.add_impala_operation(op, 'identity', 'udf_testing')

        def _identity_test(value):
            return op(value).to_expr()
        result = _identity_test('hello world')
        assert result == "SELECT udf_testing.identity('hello world')"

    def test_sql_generation_from_infoclass(self):
        udf_info = udf.UDFCreator('test.so', ['string'], 'string', 'info_test')
        op = udf_info.to_operation()
        udf.add_impala_operation(op, 'info_test', 'udf_testing')
        assert op in _operation_registry

        def _infoclass_test(value):
            return op(value).to_expr()
        result = _infoclass_test('hello world')

        assert result == "SELECT udf_testing.info_test('hello world')"

    def test_boolean_wrapping(self):
        func = self._udf_registration_single_input('boolean',
                                                   'boolean',
                                                   'test')
        expr = func(True)
        assert type(expr) == ir.BooleanScalar
        expr = func(self.b)
        assert type(expr) == ir.BooleanArray

    def test_tinyint_wrapping(self):
        func = self._udf_registration_single_input('int8',
                                                   'int8',
                                                   'test')
        expr = func(1)
        assert type(expr) == ir.Int8Scalar
        expr = func(self.i8)
        assert type(expr) == ir.Int8Array

    def test_smallint_wrapping(self):
        func = self._udf_registration_single_input('int16',
                                                   'int16',
                                                   'test')
        expr = func(1)
        assert type(expr) == ir.Int16Scalar
        expr = func(self.i16)
        assert type(expr) == ir.Int16Array

    def test_int_wrapping(self):
        func = self._udf_registration_single_input('int32',
                                                   'int32',
                                                   'test')
        expr = func(1)
        assert type(expr) == ir.Int32Scalar
        expr = func(self.i32)
        assert type(expr) == ir.Int32Array

    def test_bigint_wrapping(self):
        func = self._udf_registration_single_input('int64',
                                                   'int64',
                                                   'test')
        expr = func(1)
        assert type(expr) == ir.Int64Scalar
        expr = func(self.i64)
        assert type(expr) == ir.Int64Array

    def test_float_wrapping(self):
        func = self._udf_registration_single_input('float',
                                                   'float',
                                                   'test')
        expr = func(1.0)
        assert type(expr) == ir.FloatScalar
        expr = func(self.f)
        assert type(expr) == ir.FloatArray

    def test_double_wrapping(self):
        func = self._udf_registration_single_input('double',
                                                   'double',
                                                   'test')
        expr = func(1.0)
        assert type(expr) == ir.DoubleScalar
        expr = func(self.d)
        assert type(expr) == ir.DoubleArray

    def test_decimal_wrapping(self):
        func = self._udf_registration_single_input('decimal(9,0)',
                                                   'decimal(9,0)',
                                                   'test')
        expr = func(1.0)
        assert type(expr) == ir.DecimalScalar
        expr = func(self.dec)
        assert type(expr) == ir.DecimalArray

    def test_string_wrapping(self):
        func = self._udf_registration_single_input('string',
                                                   'string',
                                                   'test')
        expr = func('1')
        assert type(expr) == ir.StringScalar
        expr = func(self.s)
        assert type(expr) == ir.StringArray

    def test_timestamp_wrapping(self):
        func = self._udf_registration_single_input('timestamp',
                                                   'timestamp',
                                                   'test')
        expr = func(ibis.timestamp('1961-04-10'))
        assert type(expr) == ir.TimestampScalar
        expr = func(self.t)
        assert type(expr) == ir.TimestampArray

    def test_invalid_typecasting_tinyint(self):
        self._invalid_typecasts('int8', self.all_cols[1:])

    def test_invalid_typecasting_smallint(self):
        self._invalid_typecasts('int16', self.all_cols[2:])

    def test_invalid_typecasting_int(self):
        self._invalid_typecasts('int32', self.all_cols[3:])

    def test_invalid_typecasting_bigint(self):
        self._invalid_typecasts('int64', self.all_cols[4:])

    def test_invalid_typecasting_boolean(self):
        self._invalid_typecasts('boolean', self.all_cols[:8] +
                                self.all_cols[9:])

    def test_invalid_typecasting_float(self):
        self._invalid_typecasts('float', self.all_cols[:4] +
                                self.all_cols[6:])

    def test_invalid_typecasting_double(self):
        self._invalid_typecasts('double', self.all_cols[:4] +
                                self.all_cols[6:])

    def test_invalid_typecasting_string(self):
        self._invalid_typecasts('string', self.all_cols[:7] +
                                self.all_cols[8:])

    def test_invalid_typecasting_timestamp(self):
        self._invalid_typecasts('timestamp', self.all_cols[:-1])

    def test_invalid_typecasting_decimal(self):
        self._invalid_typecasts('decimal', self.all_cols[:4] +
                                self.all_cols[7:])

    def test_mult_args(self):
        op = self._udf_registration(['int32', 'double', 'string',
                                     'boolean', 'timestamp'],
                                    'int64', 'mult_types')

        def _func(integer, double, string, boolean, timestamp):
            return op(integer, double, string, boolean, timestamp).to_expr()

        expr = _func(self.i32, self.d, self.s, self.b, self.t)
        assert issubclass(type(expr), ir.ArrayExpr)

        expr = _func(1, 1.0, 'a', True, ibis.timestamp('1961-04-10'))
        assert issubclass(type(expr), ir.ScalarExpr)

    def _udf_registration_single_input(self, inputs, output, name):
        op = self._udf_registration([inputs], output, name)

        def _test_func(value):
            return op(value).to_expr()
        return _test_func

    def _udf_registration(self, inputs, output, name):
        op = udf.scalar_function(inputs, output, name=name)
        assert issubclass(op, ValueOp)
        udf.add_impala_operation(op, name, 'ibis_testing')
        return op

    def _invalid_typecasts(self, inputs, invalid_casts):
        func = self._udf_registration_single_input(inputs,
                                                   'int32',
                                                   'typecast')
        for in_type in invalid_casts:
            self.assertRaises(IbisTypeError, func, in_type)
