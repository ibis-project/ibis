from __future__ import annotations

from decimal import Decimal
from posixpath import join as pjoin

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.backends.impala as api
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util
from ibis.backends.impala import ddl
from ibis.common.annotations import ValidationError
from ibis.expr import rules

pytest.importorskip("impala")


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.fixture
def i8(table):
    return table.tinyint_col


@pytest.fixture
def i16(table):
    return table.smallint_col


@pytest.fixture
def i32(table):
    return table.int_col


@pytest.fixture
def i64(table):
    return table.bigint_col


@pytest.fixture
def d(table):
    return table.double_col


@pytest.fixture
def f(table):
    return table.float_col


@pytest.fixture
def s(table):
    return table.string_col


@pytest.fixture
def b(table):
    return table.bool_col


@pytest.fixture
def t(table):
    return table.timestamp_col


@pytest.fixture
def tpch_customer(con):
    return con.table("customer")


@pytest.fixture
def dec(tpch_customer):
    return tpch_customer.c_acctbal


@pytest.fixture
def all_cols(i8, i16, i32, i64, d, f, dec, s, b, t):
    return [
        i8,
        i16,
        i32,
        i64,
        d,
        f,
        dec,
        s,
        b,
        t,
    ]


def test_sql_generation(snapshot):
    func = api.scalar_function(["string"], "string", name="Tester")
    func.register("identity", "udf_testing")

    result = func("hello world")
    snapshot.assert_match(ibis.impala.compile(result), "out.sql")


def test_sql_generation_from_infoclass(snapshot):
    func = api.wrap_udf("test.so", ["string"], "string", "info_test")
    repr(func)

    func.register("info_test", "udf_testing")
    result = func("hello world").name("tmp")
    snapshot.assert_match(ibis.impala.compile(result), "out.sql")


@pytest.mark.parametrize(
    ("ty", "value", "column"),
    [
        pytest.param("boolean", True, "bool_col", id="boolean"),
        pytest.param("int8", 1, "tinyint_col", id="int8"),
        pytest.param("int16", 1, "smallint_col", id="int16"),
        pytest.param("int32", 1, "int_col", id="int32"),
        pytest.param("int64", 1, "bigint_col", id="int64"),
        pytest.param("float", 1.0, "float_col", id="float"),
        pytest.param("double", 1.0, "double_col", id="double"),
        pytest.param("string", "1", "string_col", id="string"),
        pytest.param(
            "timestamp",
            ibis.timestamp("1961-04-10"),
            "timestamp_col",
            id="timestamp",
        ),
    ],
)
def test_udf_primitive_output_types(ty, value, column, table):
    func = _register_udf([ty], ty, "test")

    ibis_type = dt.validate_type(ty)

    expr = func(value)
    assert type(expr) == getattr(ir, ibis_type.scalar)

    expr = func(table[column])
    assert type(expr) == getattr(ir, ibis_type.column)


@pytest.mark.parametrize(
    ("ty", "value"),
    [
        pytest.param("boolean", True, id="boolean"),
        pytest.param("int8", 1, id="int8"),
        pytest.param("int16", 1, id="int16"),
        pytest.param("int32", 1, id="int32"),
        pytest.param("int64", 1, id="int64"),
        pytest.param("float", 1.0, id="float"),
        pytest.param("double", 1.0, id="double"),
        pytest.param("string", "1", id="string"),
        pytest.param(
            "timestamp",
            ibis.timestamp("1961-04-10"),
            id="timestamp",
        ),
    ],
)
def test_uda_primitive_output_types(ty, value):
    func = _register_uda([ty], ty, "test")

    ibis_type = dt.validate_type(ty)
    scalar_type = getattr(ir, ibis_type.scalar)

    expr1 = func(value)
    assert isinstance(expr1, scalar_type)

    expr2 = func(value)
    assert isinstance(expr2, scalar_type)


def test_decimal(dec):
    func = _register_udf(["decimal(12, 2)"], "decimal(12, 2)", "test")
    expr = func(1.0)
    assert type(expr) == ir.DecimalScalar
    expr = func(dec)
    assert type(expr) == ir.DecimalColumn


@pytest.mark.parametrize(
    ("ty", "valid_cast_indexer"),
    [
        pytest.param("decimal(12, 2)", slice(7), id="decimal"),
        pytest.param("double", slice(6), id="double"),
        pytest.param("float", slice(6), id="float"),
        pytest.param("int16", slice(2), id="int16"),
        pytest.param("int32", slice(3), id="int32"),
        pytest.param("int64", slice(4), id="int64"),
        pytest.param("int8", slice(1), id="int8"),
    ],
)
def test_udf_valid_typecasting(ty, valid_cast_indexer, all_cols):
    func = _register_udf([ty], "int32", "typecast")

    for expr in all_cols[valid_cast_indexer]:
        func(expr)


@pytest.mark.parametrize(
    ("ty", "valid_cast_indexer"),
    [
        pytest.param("boolean", slice(8), id="boolean_first_8"),
        pytest.param("boolean", slice(9, None), id="boolean_9_onwards"),
        pytest.param("decimal", slice(7, None), id="decimal"),
        pytest.param("double", slice(-3, None), id="double"),
        pytest.param("float", slice(-3, None), id="float"),
        pytest.param("int16", slice(2, None), id="int16"),
        pytest.param("int32", slice(3, None), id="int32"),
        pytest.param("int64", slice(4, None), id="int64"),
        pytest.param("int8", slice(1, None), id="int8"),
        pytest.param("string", slice(7), id="string_first_7"),
        pytest.param("string", slice(8, None), id="string_8_onwards"),
        pytest.param("timestamp", slice(-1), id="timestamp"),
    ],
)
def test_udf_invalid_typecasting(ty, valid_cast_indexer, all_cols):
    func = _register_udf([ty], "int32", "typecast")

    for expr in all_cols[valid_cast_indexer]:
        with pytest.raises(ValidationError):
            func(expr)


def test_mult_args(i32, d, s, b, t):
    func = _register_udf(
        ["int32", "double", "string", "boolean", "timestamp"],
        "int64",
        "mult_types",
    )

    expr = func(i32, d, s, b, t)
    assert issubclass(type(expr), ir.Column)

    expr = func(1, 1.0, "a", True, ibis.timestamp("1961-04-10"))
    assert issubclass(type(expr), ir.Scalar)


def _register_udf(inputs, output, name):
    func = api.scalar_function(inputs, output, name=name)
    func.register(name, "ibis_testing")
    return func


def _register_uda(inputs, output, name):
    func = api.aggregate_function(inputs, output, name=name)
    func.register(name, "ibis_testing")
    return func


@pytest.fixture
def udfcon(con, monkeypatch):
    monkeypatch.setitem(con.con.options, "DISABLE_CODEGEN", "0")
    return con


@pytest.fixture
def alltypes(udfcon):
    return udfcon.table("functional_alltypes")


@pytest.fixture
def udf_ll(test_data_dir):
    return pjoin(test_data_dir, "udf/udf-sample.ll")


@pytest.fixture
def uda_ll(test_data_dir):
    return pjoin(test_data_dir, "udf/uda-sample.ll")


@pytest.fixture
def uda_so(test_data_dir):
    return pjoin(test_data_dir, "udf/libudasample.so")


@pytest.mark.parametrize(
    ("typ", "lit_val", "col_name"),
    [
        pytest.param("boolean", True, "bool_col", id="boolean"),
        pytest.param("int8", ibis.literal(5), "tinyint_col", id="int8"),
        pytest.param(
            "int16",
            ibis.literal(2**10),
            "smallint_col",
            id="int16",
        ),
        pytest.param("int32", ibis.literal(2**17), "int_col", id="int16"),
        pytest.param("int64", ibis.literal(2**33), "bigint_col", id="int64"),
        pytest.param("float", ibis.literal(3.14), "float_col", id="float"),
        pytest.param("double", ibis.literal(3.14), "double_col", id="double"),
        pytest.param(
            "string",
            ibis.literal("ibis"),
            "string_col",
            id="string",
        ),
        pytest.param(
            "timestamp",
            ibis.timestamp("1961-04-10"),
            "timestamp_col",
            id="timestamp",
        ),
    ],
)
@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_identity_primitive_types(
    udfcon, alltypes, test_data_db, udf_ll, typ, lit_val, col_name
):
    col_val = alltypes[col_name]
    identity_func_testing(udf_ll, udfcon, test_data_db, typ, lit_val, col_val)


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_decimal_fail(udfcon, test_data_db, udf_ll):
    col = udfcon.table("customer").c_acctbal
    literal = ibis.literal(1).cast("decimal(12,2)")
    name = "__tmp_udf_" + util.guid()

    func = udf_creation_to_op(
        udf_ll,
        udfcon,
        test_data_db,
        name,
        "Identity",
        ["decimal(12,2)"],
        "decimal(12,2)",
    )

    expr = func(literal)
    assert issubclass(type(expr), ir.Scalar)
    result = udfcon.execute(expr)
    assert result == Decimal(1)

    expr = func(col)
    assert issubclass(type(expr), ir.Column)
    udfcon.execute(expr)


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_mixed_inputs(udfcon, alltypes, test_data_db, udf_ll):
    name = "two_args"
    symbol = "TwoArgs"
    inputs = ["int32", "int32"]
    output = "int32"
    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, symbol, inputs, output
    )

    expr = func(alltypes.int_col, 1)
    assert issubclass(type(expr), ir.Column)
    udfcon.execute(expr)

    expr = func(1, alltypes.int_col)
    assert issubclass(type(expr), ir.Column)
    udfcon.execute(expr)

    expr = func(alltypes.int_col, alltypes.tinyint_col)
    udfcon.execute(expr)


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_implicit_typecasting(udfcon, alltypes, test_data_db, udf_ll):
    col = alltypes.tinyint_col
    literal = ibis.literal(1000)
    identity_func_testing(udf_ll, udfcon, test_data_db, "int32", literal, col)


def identity_func_testing(udf_ll, udfcon, test_data_db, datatype, literal, column):
    inputs = [datatype]
    name = "__tmp_udf_" + util.guid()
    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, "Identity", inputs, datatype
    )

    expr = func(literal)
    assert issubclass(type(expr), ir.Scalar)
    result = udfcon.execute(expr)
    # Hacky
    if datatype == "timestamp":
        assert type(result) == pd.Timestamp
    else:
        lop = literal.op()
        if isinstance(lop, ir.Literal):
            np.testing.assert_allclose(lop.value, 5)
        else:
            np.testing.assert_allclose(result, udfcon.execute(literal), 5)

    expr = func(column)
    assert issubclass(type(expr), ir.Column)
    udfcon.execute(expr)


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_mult_type_args(udfcon, alltypes, test_data_db, udf_ll):
    symbol = "AlmostAllTypes"
    name = "most_types"
    inputs = [
        "string",
        "boolean",
        "int8",
        "int16",
        "int32",
        "int64",
        "float",
        "double",
    ]
    output = "int32"

    func = udf_creation_to_op(
        udf_ll, udfcon, test_data_db, name, symbol, inputs, output
    )

    expr = func("a", True, 1, 1, 1, 1, 1.0, 1.0)
    result = udfcon.execute(expr)
    assert result == 8

    table = alltypes
    expr = func(
        table.string_col,
        table.bool_col,
        table.tinyint_col,
        table.tinyint_col,
        table.smallint_col,
        table.smallint_col,
        1.0,
        1.0,
    )
    udfcon.execute(expr)


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_udf_varargs(udfcon, alltypes, udf_ll, test_data_db):
    t = alltypes

    name = f"add_numbers_{util.guid()[:4]}"

    input_sig = rules.varargs(rules.double)
    func = api.wrap_udf(udf_ll, input_sig, "double", "AddNumbers", name=name)
    func.register(name, test_data_db)
    udfcon.create_function(func, database=test_data_db)

    expr = func(t.double_col, t.double_col)
    expr.execute()


def test_drop_udf_not_exists(udfcon):
    random_name = util.guid()
    with pytest.raises(com.MissingUDFError, match=random_name):
        udfcon.drop_udf(random_name)


def test_drop_uda_not_exists(udfcon):
    random_name = util.guid()
    with pytest.raises(com.MissingUDFError, match=random_name):
        udfcon.drop_uda(random_name)


def udf_creation_to_op(udf_ll, udfcon, test_data_db, name, symbol, inputs, output):
    func = api.wrap_udf(udf_ll, inputs, output, symbol, name)

    udfcon.create_function(func, database=test_data_db)

    func.register(name, test_data_db)

    assert udfcon.exists_udf(name, test_data_db)
    return func


def test_ll_uda_not_supported(uda_ll):
    # LLVM IR UDAs are not supported as of Impala 2.2
    with pytest.raises(com.IbisError):
        conforming_wrapper(uda_ll, ["double"], "double", "Variance")


def conforming_wrapper(where, inputs, output, prefix, serialize=True, name=None):
    kwds = {"name": name}
    if serialize:
        kwds["serialize_fn"] = f"{prefix}Serialize"
    return api.wrap_uda(
        where,
        inputs,
        output,
        f"{prefix}Update",
        init_fn=f"{prefix}Init",
        merge_fn=f"{prefix}Merge",
        finalize_fn=f"{prefix}Finalize",
        **kwds,
    )


@pytest.fixture
def wrapped_count_uda(uda_so):
    name = f"user_count_{util.guid()}"
    return api.wrap_uda(uda_so, ["int32"], "int64", "CountUpdate", name=name)


def test_count_uda(udfcon, alltypes, test_data_db, wrapped_count_uda):
    func = wrapped_count_uda
    func.register(func.name, test_data_db)
    udfcon.create_function(func, database=test_data_db)

    # it works!
    func(alltypes.int_col).execute()


def test_list_udas(udfcon, temp_database, wrapped_count_uda):
    func = wrapped_count_uda
    db = temp_database
    udfcon.create_function(func, database=db)

    funcs = udfcon.list_udas(database=db)

    f = funcs[0]
    assert f.name == func.name
    assert f.inputs == func.inputs
    assert f.output == func.output


@pytest.mark.xfail(
    reason="Unknown reason. xfailing to restore the CI for udf tests. #2358"
)
def test_drop_database_with_udfs_and_udas(udfcon, temp_database, wrapped_count_uda):
    uda1 = wrapped_count_uda

    udf1 = api.wrap_udf(
        udf_ll,
        ["boolean"],
        "boolean",
        "Identity",
        f"udf_{util.guid()}",
    )

    db = temp_database

    udfcon.create_database(db)

    udfcon.create_function(uda1, database=db)
    udfcon.create_function(udf1, database=db)
    # drop happens in test tear down


@pytest.fixture
def inputs():
    return ["string", "string"]


@pytest.fixture
def output():
    return "int64"


@pytest.fixture
def name():
    return "test_name"


def test_create_udf(inputs, output, name, snapshot):
    func = api.wrap_udf(
        "/foo/bar.so",
        inputs,
        output,
        so_symbol="testFunc",
        name=name,
    )
    stmt = ddl.CreateUDF(func)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_udf_type_conversions(output, name, snapshot):
    inputs = ["string", "int8", "int16", "int32"]
    func = api.wrap_udf(
        "/foo/bar.so",
        inputs,
        output,
        so_symbol="testFunc",
        name=name,
    )
    stmt = ddl.CreateUDF(func)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_delete_udf_simple(name, inputs, snapshot):
    stmt = ddl.DropFunction(name, inputs)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_delete_udf_if_exists(name, inputs, snapshot):
    stmt = ddl.DropFunction(name, inputs, must_exist=False)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_delete_udf_aggregate(name, inputs, snapshot):
    stmt = ddl.DropFunction(name, inputs, aggregate=True)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_delete_udf_db(name, inputs, snapshot):
    stmt = ddl.DropFunction(name, inputs, database="test")
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("series", [True, False])
def test_create_uda(name, inputs, output, series, snapshot):
    func = api.wrap_uda(
        "/foo/bar.so",
        inputs,
        output,
        update_fn="Update",
        init_fn="Init",
        merge_fn="Merge",
        finalize_fn="Finalize",
        serialize_fn="Serialize" if series else None,
    )
    stmt = ddl.CreateUDA(func, name=name, database="bar")
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_list_udf(snapshot):
    stmt = ddl.ListFunction("test")
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_list_udfs_like(snapshot):
    stmt = ddl.ListFunction("test", like="identity")
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_list_udafs(snapshot):
    stmt = ddl.ListFunction("test", aggregate=True)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_list_udafs_like(snapshot):
    stmt = ddl.ListFunction("test", like="identity", aggregate=True)
    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")
