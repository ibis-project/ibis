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

from decimal import Decimal
from posixpath import join as pjoin
import pytest

import ibis

import ibis.expr.types as ir

from ibis.impala import udf, ddl
import ibis.impala as api

from ibis.compat import unittest
from ibis.expr.datatypes import validate_type
from ibis.expr.tests.mocks import MockConnection
from ibis.sql.exprs import _operation_registry
from ibis.expr.operations import ValueOp
from ibis.common import IbisTypeError
from ibis.tests.util import ImpalaE2E
import ibis.util as util


class TestWrapping(unittest.TestCase):

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
        udf.add_operation(op, 'identity', 'udf_testing')

        def _identity_test(value):
            return op(value).to_expr()
        result = _identity_test('hello world')
        assert result == "SELECT udf_testing.identity('hello world')"

    def test_sql_generation_from_infoclass(self):
        info, op = api.wrap_udf('test.so', ['string'], 'string', 'info_test')
        repr(info)
        api.add_operation(op, 'info_test', 'udf_testing')
        assert op in _operation_registry

        def _infoclass_test(value):
            return op(value).to_expr()
        result = _infoclass_test('hello world')

        assert result == "SELECT udf_testing.info_test('hello world')"

    def test_primitive_output_types(self):
        types = [
            ('boolean', True, self.b),
            ('int8', 1, self.i8),
            ('int16', 1, self.i16),
            ('int32', 1, self.i32),
            ('int64', 1, self.i64),
            ('float', 1.0, self.f),
            ('double', 1.0, self.d),
            ('string', '1', self.s),
            ('timestamp', ibis.timestamp('1961-04-10'), self.t)
        ]
        for t, sv, av in types:
            _, func = self._register_udf([t], t, 'test')

            ibis_type = validate_type(t)

            expr = func(sv)
            assert type(expr) == ibis_type.scalar_type()
            expr = func(av)
            assert type(expr) == ibis_type.array_type()

    def test_decimal(self):
        _, func = self._register_udf(['decimal(9,0)'], 'decimal(9,0)', 'test')
        expr = func(1.0)
        assert type(expr) == ir.DecimalScalar
        expr = func(self.dec)
        assert type(expr) == ir.DecimalArray

    def test_invalid_typecasting_tinyint(self):
        cases = [
            ('int8', self.all_cols[1:]),
            ('int16', self.all_cols[2:]),
            ('int32', self.all_cols[3:]),
            ('int64', self.all_cols[4:]),
            ('boolean', self.all_cols[:8] + self.all_cols[9:]),
            ('float', self.all_cols[:4] + self.all_cols[6:]),
            ('double', self.all_cols[:4] + self.all_cols[6:]),
            ('string', self.all_cols[:7] + self.all_cols[8:]),
            ('timestamp', self.all_cols[:-1]),
            ('decimal', self.all_cols[:4] + self.all_cols[7:])
        ]

        for t, casts in cases:
            _, func = self._register_udf([t], 'int32', 'typecast')
            for in_type in casts:
                self.assertRaises(IbisTypeError, func, in_type)

    def test_mult_args(self):
        op, func = self._register_udf(['int32', 'double', 'string',
                                       'boolean', 'timestamp'],
                                      'int64', 'mult_types')

        expr = func(self.i32, self.d, self.s, self.b, self.t)
        assert issubclass(type(expr), ir.ArrayExpr)

        expr = func(1, 1.0, 'a', True, ibis.timestamp('1961-04-10'))
        assert issubclass(type(expr), ir.ScalarExpr)

    def _register_udf(self, inputs, output, name):
        op = udf.scalar_function(inputs, output, name=name)
        assert issubclass(op, ValueOp)
        udf.add_operation(op, name, 'ibis_testing')

        def func(*args):
            return op(*args).to_expr()

        return op, func

    def _register_uda(self, inputs, output, name):
        op = udf.scalar_function(inputs, output, name=name)
        assert issubclass(op, ValueOp)
        udf.add_operation(op, name, 'ibis_testing')

        def func(*args):
            return op(*args).to_expr()

        return op, func


class TestUDFE2E(ImpalaE2E, unittest.TestCase):

    def setUp(self):
        super(TestUDFE2E, self).setUp()
        self.udf_ll = pjoin(self.test_data_dir, 'udf/udf-sample.ll')
        self.uda_ll = pjoin(self.test_data_dir, 'udf/uda-sample.ll')

    @pytest.mark.udf
    def test_boolean(self):
        col = self.alltypes.bool_col
        literal = ibis.literal(True)
        self._identity_func_testing('boolean', literal, col)

    @pytest.mark.udf
    def test_tinyint(self):
        col = self.alltypes.tinyint_col
        literal = ibis.literal(5)
        self._identity_func_testing('int8', literal, col)

    @pytest.mark.udf
    def test_int(self):
        col = self.alltypes.int_col
        literal = ibis.literal(1000)
        self._identity_func_testing('int32', literal, col)

    @pytest.mark.udf
    def test_bigint(self):
        col = self.alltypes.bigint_col
        literal = ibis.literal(1000).cast('int64')
        self._identity_func_testing('int64', literal, col)

    @pytest.mark.udf
    def test_float(self):
        col = self.alltypes.float_col
        literal = ibis.literal(3.14)
        self._identity_func_testing('float', literal, col)

    @pytest.mark.udf
    def test_double(self):
        col = self.alltypes.double_col
        literal = ibis.literal(3.14)
        self._identity_func_testing('double', literal, col)

    @pytest.mark.udf
    def test_string(self):
        col = self.alltypes.string_col
        literal = ibis.literal('ibis')
        self._identity_func_testing('string', literal, col)

    @pytest.mark.udf
    def test_timestamp(self):
        col = self.alltypes.timestamp_col
        literal = ibis.timestamp('1961-04-10')
        self._identity_func_testing('timestamp', literal, col)

    @pytest.mark.udf
    def test_decimal(self):
        col = self.con.table('tpch_customer').c_acctbal
        literal = ibis.literal(1).cast('decimal(12,2)')
        name = '__tmp_udf_' + util.guid()
        op = self._udf_creation_to_op(name, 'Identity', ['decimal(12,2)'],
                                      'decimal(12,2)')

        def _func(val):
            return op(val).to_expr()
        expr = _func(literal)
        assert issubclass(type(expr), ir.ScalarExpr)
        result = self.con.execute(expr)
        assert result == Decimal(1)

        expr = _func(col)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

    @pytest.mark.udf
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

    @pytest.mark.udf
    def test_implicit_typecasting(self):
        col = self.alltypes.tinyint_col
        literal = ibis.literal(1000)
        self._identity_func_testing('int32', literal, col)

    @pytest.mark.udf
    def test_mult_type_args(self):
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

    @pytest.mark.udf
    def test_all_type_args(self):
        pytest.skip('failing test, to be fixed later')

        symbol = 'AllTypes'
        name = 'all_types'
        inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
                  'int64', 'float', 'double', 'decimal']
        output = 'int32'

        op = self._udf_creation_to_op(name, symbol, inputs, output)

        def _all_types(string, boolean, tinyint, smallint, integer,
                       bigint, float_val, double_val, decimal_val):
            return op(string, boolean, tinyint, smallint, integer,
                      bigint, float_val, double_val, decimal_val).to_expr()
        expr = _all_types('a', True, 1, 1, 1, 1, 1.0, 1.0, 1.0)
        result = self.con.execute(expr)
        assert result == 9

    @pytest.mark.udf
    def test_drop_udf_not_exists(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_udf, random_name)

    def _udf_creation_to_op(self, name, symbol, inputs, output):
        info, op = udf.wrap_udf(self.udf_ll, inputs, output, symbol, name)

        self.temp_functions.append((name, inputs))

        self.con.create_udf(info, database=self.test_data_db)
        udf.add_operation(op, name, self.test_data_db)

        assert self.con.exists_udf(name, self.test_data_db)
        return op

    def _identity_func_testing(self, datatype, literal, column):
        inputs = [datatype]
        name = '__tmp_udf_' + util.guid()
        op = self._udf_creation_to_op(name, 'Identity', inputs, datatype)

        def _identity_test(value):
            return op(value).to_expr()
        expr = _identity_test(literal)
        assert issubclass(type(expr), ir.ScalarExpr)
        result = self.con.execute(expr)
        # Hacky
        if datatype is 'timestamp':
            import pandas as pd
            assert type(result) == pd.tslib.Timestamp
        else:
            lop = literal.op()
            if isinstance(lop, ir.Literal):
                self.assertAlmostEqual(result, lop.value, 5)
            else:
                self.assertAlmostEqual(result, self.con.execute(literal), 5)

        expr = _identity_test(column)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)


class TestUDFDDL(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.name = 'test_name'
        self.inputs = ['string', 'string']
        self.output = 'int64'

    def test_create_udf(self):
        stmt = ddl.CreateFunction('/foo/bar.so', 'testFunc', self.inputs,
                                  self.output, self.name)
        result = stmt.compile()
        expected = ("CREATE FUNCTION test_name(string, string) returns bigint "
                    "location '/foo/bar.so' symbol='testFunc'")
        assert result == expected

    def test_create_udf_type_conversions(self):
        stmt = ddl.CreateFunction('/foo/bar.so', 'testFunc',
                                  ['string', 'int8', 'int16', 'int32'],
                                  self.output, self.name)
        result = stmt.compile()
        expected = ("CREATE FUNCTION test_name(string, tinyint, "
                    "smallint, int) returns bigint "
                    "location '/foo/bar.so' symbol='testFunc'")
        assert result == expected

    def test_delete_udf_simple(self):
        stmt = ddl.DropFunction(self.name, self.inputs)
        result = stmt.compile()
        expected = "DROP FUNCTION test_name(string, string)"
        assert result == expected

    def test_delete_udf_if_exists(self):
        stmt = ddl.DropFunction(self.name, self.inputs, must_exist=False)
        result = stmt.compile()
        expected = "DROP FUNCTION IF EXISTS test_name(string, string)"
        assert result == expected

    def test_delete_udf_aggregate(self):
        stmt = ddl.DropFunction(self.name, self.inputs, aggregate=True)
        result = stmt.compile()
        expected = "DROP AGGREGATE FUNCTION test_name(string, string)"
        assert result == expected

    def test_delete_udf_db(self):
        stmt = ddl.DropFunction(self.name, self.inputs, database='test')
        result = stmt.compile()
        expected = "DROP FUNCTION test.test_name(string, string)"
        assert result == expected

    def test_create_uda(self):
        stmt = ddl.CreateAggregateFunction('/foo/bar.so', self.inputs,
                                           self.output, 'Init', 'Update',
                                           'Merge', 'Finalize', self.name)
        result = stmt.compile()
        expected = ("CREATE AGGREGATE FUNCTION test_name(string, string)"
                    " returns bigint location '/foo/bar.so'"
                    " init_fn='Init' update_fn='Update'"
                    " merge_fn='Merge' finalize_fn='Finalize'")
        assert result == expected

    def test_list_udf(self):
        stmt = ddl.ListFunction('test')
        result = stmt.compile()
        expected = 'SHOW FUNCTIONS IN test'
        assert result == expected

    def test_list_udfs_like(self):
        stmt = ddl.ListFunction('test', like='identity')
        result = stmt.compile()
        expected = "SHOW FUNCTIONS IN test LIKE 'identity'"
        assert result == expected

    def test_list_udafs(self):
        stmt = ddl.ListFunction('test', aggregate=True)
        result = stmt.compile()
        expected = 'SHOW AGGREGATE FUNCTIONS IN test'
        assert result == expected

    def test_list_udafs_like(self):
        stmt = ddl.ListFunction('test', like='identity', aggregate=True)
        result = stmt.compile()
        expected = "SHOW AGGREGATE FUNCTIONS IN test LIKE 'identity'"
        assert result == expected
