from __future__ import annotations

import pytest
import sqlglot.expressions as sge

import ibis
from ibis import Deferred, _
from ibis.common import exceptions as exc
from ibis.expr import datatypes as dt
from ibis.tstring import t

my_scalar_int8 = ibis.literal(7, type="int8")
my_scalar_uint64 = ibis.literal(7, type="uint64")

unknown_dtype = dt.Unknown(raw_type=sge.DataType(this=sge.DataType.Type.UNKNOWN))


@pytest.mark.parametrize("arg_type", ["str", "template"])
@pytest.mark.parametrize(
    ("inp", "exp"),
    [
        # This is just what sqlglot infers
        ("5", dt.Int32()),
        ("'foo'", dt.String()),
        ("true", dt.Boolean()),
        ("null", unknown_dtype),
        ("'foo'::JSON", dt.JSON()),
        # This is weird that these ints are parsed as int8
        # but the plain "5" is parsed as int32?
        pytest.param(
            "{{1,2,3}}",
            dt.Array(value_type=dt.Int8()),
            marks=pytest.mark.xfail(),
        ),
        pytest.param(
            "{'x':1, 'y':'foo'}",
            dt.Struct({"x": dt.Int8(), "y": dt.String()}),
            marks=pytest.mark.xfail(),
        ),
        ("5 as my_column", dt.Int32()),
        ("null as my_column", unknown_dtype),
        # Should this error instead?
        ("x", unknown_dtype),
        ("this is not valid SQL", exc.IbisInputError),
    ],
)
def test_valueless(arg_type, inp: str, exp):
    if arg_type == "str":
        arg = inp
    elif arg_type == "template":
        arg = t(inp)
    else:
        raise AssertionError(arg_type)
    if isinstance(exp, type) and issubclass(exp, Exception):
        with pytest.raises(exp):
            ibis.sql_value(arg)
    else:
        expr = ibis.sql_value(arg)
        assert expr.type() == exp
        assert isinstance(expr, ibis.Scalar)


def test_int_simple():
    five = 5  # noqa: F841
    expr = ibis.sql_value(ibis.t("3 + {five}"))
    # Should this be int64?
    assert expr.type().is_int32()
    assert isinstance(expr, ibis.Scalar)
    op = expr.op()
    assert op.strings == ("3 + ", "")
    (val,) = op.values
    assert val.equals(ibis.literal(5).op())


def test_int_complex():
    five = 5  # noqa: F841
    expr = ibis.sql_value(ibis.t("3 + {five + 8}"))
    # Should this be int64?
    assert expr.type().is_int32()
    assert isinstance(expr, ibis.Scalar)
    op = expr.op()
    assert op.strings == ("3 + ", "")
    (val,) = op.values
    assert val.equals((ibis.literal(13)).op())


def test_literal_int_simple():
    five = ibis.literal(5)
    expr = ibis.sql_value(ibis.t("3 + {five}"))
    # Should this be int64?
    assert expr.type().is_int32()
    assert isinstance(expr, ibis.Scalar)
    op = expr.op()
    assert op.strings == ("3 + ", "")
    (val,) = op.values
    assert val.equals(five.op())


def test_literal_int_complex():
    five = ibis.literal(5)
    expr = ibis.sql_value(ibis.t("3 + {five + 8}"))
    # Should this be int64?
    assert expr.type().is_int32()
    assert isinstance(expr, ibis.Scalar)
    op = expr.op()
    assert op.strings == ("3 + ", "")
    (val,) = op.values
    assert val.equals((five + 8).op())


@pytest.mark.xfail(
    reason="Need to detect the presence of Deffereds as values in sql_value() before constructing the op"
)
def test_deferred():
    v = _.my_int_col.sum()  # noqa: F841
    expr = ibis.sql_value(ibis.t("3 + {v}"))
    assert isinstance(expr, Deferred)


def test_dialect_dtype():
    assert ibis.sql_value(ibis.t("5::DOUBLE")).type().is_float64()
    assert ibis.sql_value(ibis.t("5::DOUBLE"), dialect="duckdb").type().is_float64()
    assert ibis.sql_value(ibis.t("5::DOUBLE"), dialect="sqlite").type().is_float64()

    assert ibis.sql_value(ibis.t("5::DOUBLE"), type=int).type().is_int64()
    assert (
        ibis.sql_value(ibis.t("5::DOUBLE"), type=int, dialect="duckdb")
        .type()
        .is_int64()
    )
    assert (
        ibis.sql_value(ibis.t("5::DOUBLE"), type=int, dialect="sqlite")
        .type()
        .is_int64()
    )

    assert ibis.sql_value(ibis.t("5::REAL")).type().is_float32()
    assert ibis.sql_value(ibis.t("5::REAL"), dialect="duckdb").type().is_float32()
    assert ibis.sql_value(ibis.t("5::REAL"), dialect="sqlite").type().is_float64()

    assert ibis.sql_value(ibis.t("5::REAL"), type=int).type().is_int64()
    assert (
        ibis.sql_value(ibis.t("5::REAL"), type=int, dialect="duckdb").type().is_int64()
    )
    assert (
        ibis.sql_value(ibis.t("5::REAL"), type=int, dialect="sqlite").type().is_int64()
    )


def test_multiple_relations():
    t1 = ibis.table({"i": int})  # noqa: F841
    t2 = ibis.table({"i": int})  # noqa: F841
    with pytest.raises(exc.IbisInputError):
        ibis.sql_value(ibis.t("{t1.i} + {t2.i}"))
    with pytest.raises(exc.IbisInputError):
        ibis.sql_value(ibis.t("{t1.i + t2.i}"))
