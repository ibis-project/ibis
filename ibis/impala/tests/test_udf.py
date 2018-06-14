import unittest

from posixpath import join as pjoin

import pytest

import numpy as np
import pandas as pd

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.rules as rules
import ibis.common as com
import ibis.util as util

from ibis.common import IbisTypeError
from ibis.compat import Decimal
from ibis.expr.tests.mocks import MockConnection

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

pytestmark = [pytest.mark.impala, pytest.mark.udf]

from ibis.impala import ddl  # noqa: E402
import ibis.impala as api  # noqa: E402


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

            ibis_type = dt.validate_type(t)

            expr = func(sv)
            assert type(expr) == type(ibis_type.scalar_type()(expr.op()))  # noqa: E501, E721
            expr = func(av)
            assert type(expr) == type(ibis_type.array_type()(expr.op()))  # noqa: E501, E721

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

            ibis_type = dt.validate_type(t)

            expr1 = func(sv)
            expr2 = func(sv)
            expected_type1 = type(ibis_type.scalar_type()(expr1.op()))
            expected_type2 = type(ibis_type.scalar_type()(expr2.op()))
            assert isinstance(expr1, expected_type1)
            assert isinstance(expr2, expected_type2)

    def test_decimal(self):
        func = self._register_udf(['decimal(9,0)'], 'decimal(9,0)', 'test')
        expr = func(1.0)
        assert type(expr) == ir.DecimalScalar
        expr = func(self.dec)
        assert type(expr) == ir.DecimalColumn

    def test_udf_invalid_typecasting(self):
        cases = [
            ('int8', self.all_cols[:1], self.all_cols[1:]),
            ('int16', self.all_cols[:2], self.all_cols[2:]),
            ('int32', self.all_cols[:3], self.all_cols[3:]),
            ('int64', self.all_cols[:4], self.all_cols[4:]),
            ('boolean', [], self.all_cols[:8] + self.all_cols[9:]),

            # allowing double here for now
            ('float', self.all_cols[:6], [self.s, self.b, self.t]),

            ('double', self.all_cols[:6], [self.s, self.b, self.t]),
            ('string', [], self.all_cols[:7] + self.all_cols[8:]),
            ('timestamp', [], self.all_cols[:-1]),
            ('decimal', self.all_cols[:7], self.all_cols[7:])
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
        assert issubclass(type(expr), ir.ColumnExpr)

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


@pytest.fixture(scope='session')
def udfcon(con):
    con.disable_codegen(False)
    try:
        yield con
    finally:
        con.disable_codegen(True)


@pytest.fixture(scope='session')
def alltypes(udfcon):
    return udfcon.table('functional_alltypes')


@pytest.fixture(scope='session')
def udf_ll(udfcon, test_data_dir):
    return pjoin(test_data_dir, 'udf/udf-sample.ll')


@pytest.fixture(scope='session')
def uda_ll(udfcon, test_data_dir):
    return pjoin(test_data_dir, 'udf/uda-sample.ll')


@pytest.fixture(scope='session')
def uda_so(udfcon, test_data_dir):
    return pjoin(test_data_dir, 'udf/libudasample.so')


@pytest.mark.parametrize(
    ('typ', 'lit_val', 'col_name'),
    [
        ('boolean', True, 'bool_col'),
        ('int8', ibis.literal(5), 'tinyint_col'),
        ('int16', ibis.literal(2**10), 'smallint_col'),
        ('int32', ibis.literal(2**17), 'int_col'),
        ('int64', ibis.literal(2**33), 'bigint_col'),
        ('float', ibis.literal(3.14), 'float_col'),
        ('double', ibis.literal(3.14), 'double_col'),
        ('string', ibis.literal('ibis'), 'string_col'),
        ('timestamp', ibis.timestamp('1961-04-10'), 'timestamp_col'),
    ]
)
def test_identity_primitive_types(
    udfcon, alltypes, test_data_db, udf_ll, typ, lit_val, col_name
):
    col_val = alltypes[col_name]
    identity_func_testing(udf_ll, udfcon, test_data_db, typ, lit_val, col_val)


def test_decimal(udfcon, test_data_db, udf_ll):
    col = udfcon.table('tpch_customer').c_acctbal
    literal = ibis.literal(1).cast('decimal(12,2)')
    name = '__tmp_udf_' + util.guid()

    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, 'Identity',
        ['decimal(12,2)'], 'decimal(12,2)')

    expr = func(literal)
    assert issubclass(type(expr), ir.ScalarExpr)
    result = udfcon.execute(expr)
    assert result == Decimal(1)

    expr = func(col)
    assert issubclass(type(expr), ir.ColumnExpr)
    udfcon.execute(expr)


def test_mixed_inputs(udfcon, alltypes, test_data_db, udf_ll):
    name = 'two_args'
    symbol = 'TwoArgs'
    inputs = ['int32', 'int32']
    output = 'int32'
    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, symbol, inputs, output)

    expr = func(alltypes.int_col, 1)
    assert issubclass(type(expr), ir.ColumnExpr)
    udfcon.execute(expr)

    expr = func(1, alltypes.int_col)
    assert issubclass(type(expr), ir.ColumnExpr)
    udfcon.execute(expr)

    expr = func(alltypes.int_col, alltypes.tinyint_col)
    udfcon.execute(expr)


def test_implicit_typecasting(udfcon, alltypes, test_data_db, udf_ll):
    col = alltypes.tinyint_col
    literal = ibis.literal(1000)
    identity_func_testing(udf_ll, udfcon, test_data_db, 'int32', literal, col)


def identity_func_testing(
    udf_ll, udfcon, test_data_db, datatype, literal, column
):
    inputs = [datatype]
    name = '__tmp_udf_' + util.guid()
    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, 'Identity', inputs, datatype)

    expr = func(literal)
    assert issubclass(type(expr), ir.ScalarExpr)
    result = udfcon.execute(expr)
    # Hacky
    if datatype is 'timestamp':
        assert type(result) == pd.Timestamp
    else:
        lop = literal.op()
        if isinstance(lop, ir.Literal):
            np.testing.assert_allclose(lop.value, 5)
        else:
            np.testing.assert_allclose(result, udfcon.execute(literal), 5)

    expr = func(column)
    assert issubclass(type(expr), ir.ColumnExpr)
    udfcon.execute(expr)


def test_mult_type_args(udfcon, alltypes, test_data_db, udf_ll):
    symbol = 'AlmostAllTypes'
    name = 'most_types'
    inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
              'int64', 'float', 'double']
    output = 'int32'

    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, symbol, inputs, output)

    expr = func('a', True, 1, 1, 1, 1, 1.0, 1.0)
    result = udfcon.execute(expr)
    assert result == 8

    table = alltypes
    expr = func(table.string_col, table.bool_col, table.tinyint_col,
                table.tinyint_col, table.smallint_col,
                table.smallint_col, 1.0, 1.0)
    udfcon.execute(expr)


def test_all_type_args(udfcon, test_data_db, udf_ll):
    pytest.skip('failing test, to be fixed later')

    symbol = 'AllTypes'
    name = 'all_types'
    inputs = ['string', 'boolean', 'int8', 'int16', 'int32',
              'int64', 'float', 'double', 'decimal']
    output = 'int32'

    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, symbol, inputs, output)
    expr = func('a', True, 1, 1, 1, 1, 1.0, 1.0, 1.0)
    result = udfcon.execute(expr)
    assert result == 9


def test_udf_varargs(udfcon, alltypes, udf_ll, test_data_db):
    t = alltypes

    name = 'add_numbers_{0}'.format(util.guid()[:4])

    input_sig = rules.varargs(rules.double)
    func = api.wrap_udf(udf_ll, input_sig, 'double', 'AddNumbers',
                        name=name)
    func.register(name, test_data_db)
    udfcon.create_function(func, database=test_data_db)

    expr = func(t.double_col, t.double_col)
    expr.execute()


def test_drop_udf_not_exists(udfcon):
    random_name = util.guid()
    with pytest.raises(Exception):
        udfcon.drop_udf(random_name)


def test_drop_uda_not_exists(udfcon):
    random_name = util.guid()
    with pytest.raises(Exception):
        udfcon.drop_uda(random_name)


def udf_creation_to_op(
    udf_ll, udfcon, test_data_db, name, symbol, inputs, output
):
    func = api.wrap_udf(udf_ll, inputs, output, symbol, name)

    # self.temp_udfs.append((name, inputs))

    udfcon.create_function(func, database=test_data_db)

    func.register(name, test_data_db)

    assert udfcon.exists_udf(name, test_data_db)
    return func


def test_ll_uda_not_supported(uda_ll):
    # LLVM IR UDAs are not supported as of Impala 2.2
    with pytest.raises(com.IbisError):
        conforming_wrapper(uda_ll, ['double'], 'double', 'Variance')


def conforming_wrapper(
    where, inputs, output, prefix, serialize=True, name=None
):
    kwds = {'name': name}
    if serialize:
        kwds['serialize_fn'] = '{0}Serialize'.format(prefix)
    return api.wrap_uda(where, inputs, output, '{0}Update'.format(prefix),
                        init_fn='{0}Init'.format(prefix),
                        merge_fn='{0}Merge'.format(prefix),
                        finalize_fn='{0}Finalize'.format(prefix),
                        **kwds)


@pytest.fixture
def wrapped_count_uda(uda_so):
    name = 'user_count_{0}'.format(util.guid())
    return api.wrap_uda(uda_so, ['int32'], 'int64', 'CountUpdate', name=name)


def test_count_uda(udfcon, alltypes, test_data_db, wrapped_count_uda):
    func = wrapped_count_uda
    func.register(func.name, test_data_db)
    udfcon.create_function(func, database=test_data_db)

    # it works!
    func(alltypes.int_col).execute()
    # self.temp_udas.append((func.name, ['int32']))


def test_list_udas(udfcon, temp_database, wrapped_count_uda):
    func = wrapped_count_uda
    db = temp_database
    udfcon.create_function(func, database=db)

    funcs = udfcon.list_udas(database=db)

    f = funcs[0]
    assert f.name == func.name
    assert f.inputs == func.inputs
    assert f.output == func.output


def test_drop_database_with_udfs_and_udas(
    udfcon, temp_database, wrapped_count_uda
):
    uda1 = wrapped_count_uda

    udf1 = api.wrap_udf(udf_ll, ['boolean'], 'boolean', 'Identity',
                        'udf_{0}'.format(util.guid()))

    db = temp_database

    udfcon.create_database(db)

    udfcon.create_function(uda1, database=db)
    udfcon.create_function(udf1, database=db)
    # drop happens in test tear down


class TestUDFDDL(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.name = 'test_name'
        self.inputs = ['string', 'string']
        self.output = 'int64'

    def test_create_udf(self):
        func = api.wrap_udf('/foo/bar.so', self.inputs, self.output,
                            so_symbol='testFunc', name=self.name)
        stmt = ddl.CreateUDF(func)
        result = stmt.compile()
        expected = ("CREATE FUNCTION `test_name`(string, string) "
                    "returns bigint "
                    "location '/foo/bar.so' symbol='testFunc'")
        assert result == expected

    def test_create_udf_type_conversions(self):
        inputs = ['string', 'int8', 'int16', 'int32']
        func = api.wrap_udf('/foo/bar.so', inputs, self.output,
                            so_symbol='testFunc', name=self.name)
        stmt = ddl.CreateUDF(func)

        # stmt = ddl.CreateFunction('/foo/bar.so', 'testFunc',
        #                           ,
        #                           self.output, self.name)
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
            func = api.wrap_uda('/foo/bar.so', self.inputs, self.output,
                                update_fn='Update', init_fn='Init',
                                merge_fn='Merge', finalize_fn='Finalize',
                                serialize_fn='Serialize' if ser else None)
            stmt = ddl.CreateUDA(func, name=self.name, database='bar')
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
