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

from posixpath import join as pjoin
import pytest

import ibis

import ibis.expr.types as ir

from ibis.impala import ddl
import ibis.impala as api

from ibis.common import IbisTypeError
from ibis.compat import unittest, Decimal
from ibis.expr.datatypes import validate_type
from ibis.expr.tests.mocks import MockConnection
from ibis.impala.tests.common import ImpalaE2E
import ibis.expr.rules as rules
import ibis.common as com
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
        func = api.scalar_function(['string'], 'string', name='Tester')
        func.register('identity', 'udf_testing')

        result = func('hello world')
        assert (ibis.impala.compile(result) ==
                "SELECT udf_testing.identity('hello world') AS `tmp`")

    def test_sql_generation_from_infoclass(self):
        func = api.wrap_udf('test.so', ['string'], 'string', 'info_test')
        repr(func)

        func.register('info_test', 'udf_testing')
        result = func('hello world')
        assert (ibis.impala.compile(result) ==
                "SELECT udf_testing.info_test('hello world') AS `tmp`")

    def test_udf_primitive_output_types(self):
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
            func = self._register_udf([t], t, 'test')

            ibis_type = validate_type(t)

            expr = func(sv)
            assert type(expr) == ibis_type.scalar_type()
            expr = func(av)
            assert type(expr) == ibis_type.array_type()

    def test_uda_primitive_output_types(self):
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
            func = self._register_uda([t], t, 'test')

            ibis_type = validate_type(t)

            expr1 = func(sv)
            expr2 = func(sv)
            assert isinstance(expr1, ibis_type.scalar_type())
            assert isinstance(expr2, ibis_type.scalar_type())

    def test_decimal(self):
        func = self._register_udf(['decimal(9,0)'], 'decimal(9,0)', 'test')
        expr = func(1.0)
        assert type(expr) == ir.DecimalScalar
        expr = func(self.dec)
        assert type(expr) == ir.DecimalArray

    def test_udf_invalid_typecasting(self):
        cases = [
            ('int8', self.all_cols[:1], self.all_cols[1:]),
            ('int16', self.all_cols[:2], self.all_cols[2:]),
            ('int32', self.all_cols[:3], self.all_cols[3:]),
            ('int64', self.all_cols[:4], self.all_cols[4:]),
            ('boolean', [], self.all_cols[:8] + self.all_cols[9:]),

            # allowing double here for now
            ('float', self.all_cols[:4], [self.s, self.b, self.t, self.dec]),

            ('double', self.all_cols[:4], [self.s, self.b, self.t, self.dec]),
            ('string', [], self.all_cols[:7] + self.all_cols[8:]),
            ('timestamp', [], self.all_cols[:-1]),
            ('decimal', [], self.all_cols[:4] + self.all_cols[7:])
        ]

        for t, valid_casts, invalid_casts in cases:
            func = self._register_udf([t], 'int32', 'typecast')

            for expr in valid_casts:
                func(expr)

            for expr in invalid_casts:
                self.assertRaises(IbisTypeError, func, expr)

    def test_mult_args(self):
        func = self._register_udf(['int32', 'double', 'string',
                                   'boolean', 'timestamp'],
                                  'int64', 'mult_types')

        expr = func(self.i32, self.d, self.s, self.b, self.t)
        assert issubclass(type(expr), ir.ArrayExpr)

        expr = func(1, 1.0, 'a', True, ibis.timestamp('1961-04-10'))
        assert issubclass(type(expr), ir.ScalarExpr)

    def _register_udf(self, inputs, output, name):
        func = api.scalar_function(inputs, output, name=name)
        func.register(name, 'ibis_testing')
        return func

    def _register_uda(self, inputs, output, name):
        func = api.aggregate_function(inputs, output, name=name)
        func.register(name, 'ibis_testing')
        return func


class TestUDFE2E(ImpalaE2E, unittest.TestCase):

    def setUp(self):
        super(TestUDFE2E, self).setUp()
        self.udf_ll = pjoin(self.test_data_dir, 'udf/udf-sample.ll')
        self.uda_ll = pjoin(self.test_data_dir, 'udf/uda-sample.ll')
        self.uda_so = pjoin(self.test_data_dir, 'udf/libudasample.so')

    @pytest.mark.udf
    def test_identity_primitive_types(self):
        cases = [
            ('boolean', True, self.alltypes.bool_col),
            ('int8', 5, self.alltypes.tinyint_col),
            ('int16', 2**10, self.alltypes.smallint_col),
            ('int32', 2**17, self.alltypes.int_col),
            ('int64', 2**33, self.alltypes.bigint_col),
            ('float', 3.14, self.alltypes.float_col),
            ('double', 3.14, self.alltypes.double_col),
            ('string', 'ibis', self.alltypes.string_col),
            ('timestamp', ibis.timestamp('1961-04-10'),
             self.alltypes.timestamp_col),
        ]

        for t, lit_val, array_val in cases:
            if not isinstance(lit_val, ir.Expr):
                lit_val = ibis.literal(lit_val)
            self._identity_func_testing(t, lit_val, array_val)

    @pytest.mark.udf
    def test_decimal(self):
        col = self.con.table('tpch_customer').c_acctbal
        literal = ibis.literal(1).cast('decimal(12,2)')
        name = '__tmp_udf_' + util.guid()
        func = self._udf_creation_to_op(name, 'Identity',
                                        ['decimal(12,2)'],
                                        'decimal(12,2)')

        expr = func(literal)
        assert issubclass(type(expr), ir.ScalarExpr)
        result = self.con.execute(expr)
        assert result == Decimal(1)

        expr = func(col)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

    @pytest.mark.udf
    def test_mixed_inputs(self):
        name = 'two_args'
        symbol = 'TwoArgs'
        inputs = ['int32', 'int32']
        output = 'int32'
        func = self._udf_creation_to_op(name, symbol, inputs, output)

        expr = func(self.alltypes.int_col, 1)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

        expr = func(1, self.alltypes.int_col)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

        expr = func(self.alltypes.int_col, self.alltypes.tinyint_col)
        self.con.execute(expr)

    @pytest.mark.udf
    def test_implicit_typecasting(self):
        col = self.alltypes.tinyint_col
        literal = ibis.literal(1000)
        self._identity_func_testing('int32', literal, col)

    def _identity_func_testing(self, datatype, literal, column):
        inputs = [datatype]
        name = '__tmp_udf_' + util.guid()
        func = self._udf_creation_to_op(name, 'Identity', inputs, datatype)

        expr = func(literal)
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

        expr = func(column)
        assert issubclass(type(expr), ir.ArrayExpr)
        self.con.execute(expr)

    @pytest.mark.udf
    def test_mult_type_args(self):
        symbol = 'AlmostAllTypes'
        name = 'most_types'
        inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
                  'int64', 'float', 'double']
        output = 'int32'

        func = self._udf_creation_to_op(name, symbol, inputs, output)

        expr = func('a', True, 1, 1, 1, 1, 1.0, 1.0)
        result = self.con.execute(expr)
        assert result == 8

        table = self.alltypes
        expr = func(table.string_col, table.bool_col, table.tinyint_col,
                    table.tinyint_col, table.smallint_col,
                    table.smallint_col, 1.0, 1.0)
        self.con.execute(expr)

    def test_all_type_args(self):
        pytest.skip('failing test, to be fixed later')

        symbol = 'AllTypes'
        name = 'all_types'
        inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
                  'int64', 'float', 'double', 'decimal']
        output = 'int32'

        func = self._udf_creation_to_op(name, symbol, inputs, output)
        expr = func('a', True, 1, 1, 1, 1, 1.0, 1.0, 1.0)
        result = self.con.execute(expr)
        assert result == 9

    @pytest.mark.udf
    def test_udf_varargs(self):
        t = self.alltypes

        name = 'add_numbers_{0}'.format(util.guid()[:4])

        input_sig = rules.varargs(rules.double)
        func = api.wrap_udf(self.udf_ll, input_sig, 'double', 'AddNumbers',
                            name=name)
        func.register(name, self.test_data_db)
        self.con.create_function(func, database=self.test_data_db)

        expr = func(t.double_col, t.double_col)
        expr.execute()

    def test_drop_udf_not_exists(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_udf, random_name)

    def test_drop_uda_not_exists(self):
        random_name = util.guid()
        self.assertRaises(Exception, self.con.drop_uda, random_name)

    def _udf_creation_to_op(self, name, symbol, inputs, output):
        func = api.wrap_udf(self.udf_ll, inputs, output, symbol, name)

        self.temp_udfs.append((name, inputs))

        self.con.create_function(func, database=self.test_data_db)

        func.register(name, self.test_data_db)

        assert self.con.exists_udf(name, self.test_data_db)
        return func

    def test_ll_uda_not_supported(self):
        # LLVM IR UDAs are not supported as of Impala 2.2
        with self.assertRaises(com.IbisError):
            self._conforming_wrapper(self.uda_ll, ['double'], 'double',
                                     'Variance')

    def _conforming_wrapper(self, where, inputs, output, prefix,
                            serialize=True, name=None):
        kwds = {
            'name': name
        }
        if serialize:
            kwds['serialize_fn'] = '{0}Serialize'.format(prefix)
        return api.wrap_uda(where, inputs, output, '{0}Update'.format(prefix),
                            init_fn='{0}Init'.format(prefix),
                            merge_fn='{0}Merge'.format(prefix),
                            finalize_fn='{0}Finalize'.format(prefix),
                            **kwds)

    @pytest.mark.udf
    def test_count_uda(self):
        func = self._wrap_count_uda()
        func.register(func.name, self.test_data_db)
        self.con.create_function(func, database=self.test_data_db)

        # it works!
        func(self.alltypes.int_col).execute()
        self.temp_udas.append((func.name, ['int32']))

    @pytest.mark.udf
    def test_list_udas(self):
        db = '__ibis_tmp_{0}'.format(util.guid())
        self.con.create_database(db)
        self.temp_databases.append(db)

        func = self._wrap_count_uda()
        self.con.create_function(func, database=db)

        funcs = self.con.list_udas(database=db)

        f = funcs[0]
        assert f.name == func.name
        assert f.inputs == func.inputs
        assert f.output == func.output

    @pytest.mark.udf
    def test_drop_database_with_udfs_and_udas(self):
        uda1 = self._wrap_count_uda()
        uda2 = self._wrap_count_uda()

        udf1 = api.wrap_udf(self.udf_ll, ['boolean'], 'boolean', 'Identity',
                            'udf_{0}'.format(util.guid()))

        db = '__ibis_tmp_{0}'.format(util.guid())

        self.con.create_database(db)

        self.con.create_function(uda1, database=db)
        self.con.create_function(uda2, database=db)

        self.con.create_function(udf1, database=db)

        self.con.drop_database(db, force=True)

        assert not self.con.exists_database(db)

    def _wrap_count_uda(self, name=None):
        if name is None:
            name = 'user_count_{0}'.format(util.guid())
        func = api.wrap_uda(self.uda_so, ['int32'], 'int64',
                            'CountUpdate', name=name)
        return func


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
        expected = ("CREATE FUNCTION `test_name`(string, string) "
                    "returns bigint "
                    "location '/foo/bar.so' symbol='testFunc'")
        assert result == expected

    def test_create_udf_type_conversions(self):
        stmt = ddl.CreateFunction('/foo/bar.so', 'testFunc',
                                  ['string', 'int8', 'int16', 'int32'],
                                  self.output, self.name)
        result = stmt.compile()
        expected = ("CREATE FUNCTION `test_name`(string, tinyint, "
                    "smallint, int) returns bigint "
                    "location '/foo/bar.so' symbol='testFunc'")
        assert result == expected

    def test_delete_udf_simple(self):
        stmt = ddl.DropFunction(self.name, self.inputs)
        result = stmt.compile()
        expected = "DROP FUNCTION `test_name`(string, string)"
        assert result == expected

    def test_delete_udf_if_exists(self):
        stmt = ddl.DropFunction(self.name, self.inputs, must_exist=False)
        result = stmt.compile()
        expected = "DROP FUNCTION IF EXISTS `test_name`(string, string)"
        assert result == expected

    def test_delete_udf_aggregate(self):
        stmt = ddl.DropFunction(self.name, self.inputs, aggregate=True)
        result = stmt.compile()
        expected = "DROP AGGREGATE FUNCTION `test_name`(string, string)"
        assert result == expected

    def test_delete_udf_db(self):
        stmt = ddl.DropFunction(self.name, self.inputs, database='test')
        result = stmt.compile()
        expected = "DROP FUNCTION test.`test_name`(string, string)"
        assert result == expected

    def test_create_uda(self):
        def make_ex(serialize=False):
            if serialize:
                serialize = "\nserialize_fn='Serialize'"
            else:
                serialize = ""
            return (("CREATE AGGREGATE FUNCTION "
                     "bar.`test_name`(string, string)"
                     " returns bigint location '/foo/bar.so'"
                     "\ninit_fn='Init'"
                     "\nupdate_fn='Update'"
                     "\nmerge_fn='Merge'") +
                    serialize +
                    ("\nfinalize_fn='Finalize'"))

        for ser in [True, False]:
            stmt = ddl.CreateAggregateFunction('/foo/bar.so', self.inputs,
                                               self.output, 'Update', 'Init',
                                               'Merge',
                                               'Serialize' if ser else None,
                                               'Finalize', self.name, 'bar')
            result = stmt.compile()
            expected = make_ex(ser)
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
