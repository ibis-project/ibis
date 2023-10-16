from __future__ import annotations

import functools
import ipaddress
import operator
import uuid
from collections import OrderedDict
from datetime import date, datetime, time
from decimal import Decimal
from operator import attrgetter, methodcaller

import numpy as np
import pytest
import pytz
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _, literal
from ibis.common.annotations import ValidationError
from ibis.common.collections import frozendict
from ibis.common.exceptions import IbisTypeError
from ibis.expr import api
from ibis.tests.util import assert_equal


def test_null():
    assert ibis.literal(None).equals(ibis.null())
    assert ibis.null().op() == ops.Literal(None, dtype=dt.null)


@pytest.mark.parametrize(
    ["value", "expected_type"],
    [
        (5, "int8"),
        (127, "int8"),
        (128, "int16"),
        (32767, "int16"),
        (32768, "int32"),
        (2147483647, "int32"),
        (2147483648, "int64"),
        (-5, "int8"),
        (-128, "int8"),
        (-129, "int16"),
        (-32769, "int32"),
        (-2147483649, "int64"),
        (1.5, "float64"),
        ("foo", "string"),
        (b"fooblob", "bytes"),
        ([1, 2, 3], "array<int8>"),
        (ipaddress.ip_address("1.2.3.4"), "inet"),
        (ipaddress.ip_address("::1"), "inet"),
    ],
)
def test_literal_with_implicit_type(value, expected_type):
    expr = ibis.literal(value)

    assert isinstance(expr, ir.Scalar)
    assert expr.type() == dt.dtype(expected_type)

    assert isinstance(expr.op(), ops.Literal)
    assert expr.op().value == dt.normalize(expr.type(), value)


@pytest.mark.parametrize(
    ["value", "expected_type", "expected_value"],
    [
        ([1, 2, 3], "array<int8>", (1, 2, 3)),
        ([[1, 2], [3, 4]], "array<array<int8>>", ((1, 2), (3, 4))),
    ],
)
def test_listeral_with_unhashable_values(value, expected_type, expected_value):
    expr = ibis.literal(value)

    assert isinstance(expr, ir.Scalar)
    assert expr.type() == dt.dtype(expected_type)

    assert isinstance(expr.op(), ops.Literal)
    assert expr.op().value == expected_value


pointA = (1, 2)
pointB = (-3, 4)
pointC = (5, 19)
lineAB = [pointA, pointB]
lineBC = [pointB, pointC]
lineCA = [pointC, pointA]
polygon1 = [lineAB, lineBC, lineCA]
polygon2 = [lineAB, lineBC, lineCA]
multilinestring = [lineAB, lineBC, lineCA]
multipoint = [pointA, pointB, pointC]
multipolygon1 = [polygon1, polygon2]


@pytest.mark.parametrize(
    ["value", "expected_type"],
    [
        (5, "int16"),
        (127, "float64"),
        (128, "int64"),
        (32767, "float64"),
        (32768, "float32"),
        (2147483647, "int64"),
        (-5, "int16"),
        (-128, "int32"),
        (-129, "int64"),
        (-32769, "float32"),
        (-2147483649, "float64"),
        (1.5, "float64"),
        ("foo", "string"),
        (ipaddress.ip_address("1.2.3.4"), "inet"),
        (ipaddress.ip_address("::1"), "inet"),
        (list(pointA), "point"),
        (tuple(pointA), "point"),
        (list(lineAB), "linestring"),
        (tuple(lineAB), "linestring"),
        (list(polygon1), "polygon"),
        (tuple(polygon1), "polygon"),
        (list(multilinestring), "multilinestring"),
        (tuple(multilinestring), "multilinestring"),
        (list(multipoint), "multipoint"),
        (tuple(multipoint), "multipoint"),
        (list(multipolygon1), "multipolygon"),
        (tuple(multipolygon1), "multipolygon"),
        param(uuid.uuid4(), "uuid", id="uuid"),
        param(str(uuid.uuid4()), "uuid", id="uuid_str"),
        param(Decimal("234.234"), "decimal(6, 3)", id="decimal_native"),
        param(234234, "decimal(9, 3)", id="decimal_int"),
    ],
)
def test_literal_with_explicit_type(value, expected_type):
    expr = ibis.literal(value, type=expected_type)
    assert expr.type().equals(dt.validate_type(expected_type))


@pytest.mark.parametrize(
    ("value", "expected", "dtype"),
    [
        # precision > scale
        (Decimal("234.234"), Decimal("234.234"), "decimal(6, 3)"),
        (234234, Decimal("234234.000"), "decimal(9, 3)"),
        # scale == 0
        (Decimal("234"), Decimal("234"), "decimal(6, 0)"),
        (234, Decimal("234"), "decimal(6, 0)"),
        # precision == scale
        (Decimal(".234"), Decimal(".234"), "decimal(3, 3)"),
        (234, Decimal("234.000"), "decimal(6, 3)"),
    ],
)
def test_normalize_decimal_literal(value, expected, dtype):
    expr = ibis.literal(value, type=dtype)
    assert expr.op().value == expected


@pytest.mark.parametrize(
    ["value", "expected_type", "expected_class"],
    [
        (list("abc"), "array<string>", ir.ArrayScalar),
        ([1, 2, 3], "array<int8>", ir.ArrayScalar),
        ({"a": 1, "b": 2, "c": 3}, "map<string, int8>", ir.MapScalar),
        ({1: 2, 3: 4, 5: 6}, "map<int8, int8>", ir.MapScalar),
        (
            {"a": [1.0, 2.0], "b": [], "c": [3.0]},
            "map<string, array<double>>",
            ir.MapScalar,
        ),
        (
            OrderedDict(
                [
                    ("a", 1),
                    ("b", list("abc")),
                    ("c", OrderedDict([("foo", [1.0, 2.0])])),
                ]
            ),
            "struct<a: int8, b: array<string>, c: struct<foo: array<double>>>",
            ir.StructScalar,
        ),
    ],
)
def test_literal_complex_types(value, expected_type, expected_class):
    expr = ibis.literal(value)
    expr_type = expr.type()
    assert expr_type.equals(dt.validate_type(expected_type))
    assert isinstance(expr, expected_class)
    assert isinstance(expr.op(), ops.Literal)
    assert expr.op().value == dt.normalize(dt.dtype(expected_type), value)


def test_simple_map_operations():
    value = {"a": [1.0, 2.0], "b": [], "c": [3.0]}
    value2 = {"a": [1.0, 2.0], "c": [3.0], "d": [4.0, 5.0]}
    expr = ibis.literal(value)
    expr2 = ibis.literal(value2)
    assert isinstance(expr, ir.MapValue)
    assert isinstance(expr.length().op(), ops.MapLength)
    assert isinstance((expr + expr2).op(), ops.MapMerge)
    assert isinstance((expr2 + expr).op(), ops.MapMerge)

    default = ibis.literal([0.0])
    assert isinstance(expr.get("d", default).op(), ops.MapGet)

    # test for an invalid default type, nulls are ok
    with pytest.raises(IbisTypeError):
        expr.get("d", ibis.literal("foo"))

    assert isinstance(expr.get("d", ibis.null()).op(), ops.MapGet)

    assert isinstance(expr["b"].op(), ops.MapGet)
    assert isinstance(expr.keys().op(), ops.MapKeys)
    assert isinstance(expr.values().op(), ops.MapValues)


@pytest.mark.parametrize(
    ["value", "expected_type"],
    [
        (32767, "int8"),
        (32768, "int16"),
        (2147483647, "int16"),
        (2147483648, "int32"),
    ],
)
def test_literal_with_non_coercible_type(value, expected_type):
    with pytest.raises(TypeError, match="out of bounds"):
        ibis.literal(value, type=expected_type)


def test_literal_double_from_string_fails():
    with pytest.raises(TypeError, match="Unable to normalize"):
        ibis.literal("foo", type="double")


def test_list_and_tuple_literals():
    what = [1, 2, 1000]
    expr = api.literal(what)
    assert isinstance(expr, ir.ArrayScalar)
    # it works!
    repr(expr)

    what = (1, 2, 1000)
    expr = api.literal(what)
    assert isinstance(expr, ir.ArrayScalar)
    # it works!
    repr(expr)

    # test using explicit type
    point = ibis.literal((1, 2, 1000), type="point")
    assert point.type() == dt.point
    point = ibis.literal([1, 2, 1000], type="point")
    assert point.type() == dt.point


def test_literal_array():
    what = []
    expr = api.literal(what)
    assert isinstance(expr, ir.ArrayValue)
    assert expr.type().equals(dt.Array(dt.null))


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_isin_notin_list(table, container):
    values = container([1, 2, 3, 4])

    expr = table.a.isin(values)
    not_expr = table.a.notin(values)

    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.InValues)

    assert isinstance(not_expr, ir.BooleanColumn)
    assert isinstance(not_expr.op(), ops.Not)
    assert isinstance(not_expr.op().arg, ops.InValues)


def test_value_counts(table, string_col):
    bool_clause = table[string_col].notin(["1", "4", "7"])
    expr = table[bool_clause][string_col].value_counts()
    assert isinstance(expr, ir.Table)


def test_isin_notin_scalars():
    a, b, c = (ibis.literal(x) for x in [1, 1, 2])

    result = a.isin([1, 2])
    assert isinstance(result, ir.BooleanScalar)

    result = a.notin([b, c, 3])
    assert isinstance(result, ir.BooleanScalar)


def test_scalar_isin_list_with_array(table):
    val = ibis.literal(2)

    options = [table.a, table.b, table.c]

    expr = val.isin(options)
    assert isinstance(expr, ir.BooleanColumn)

    not_expr = val.notin(options)
    assert isinstance(not_expr, ir.BooleanColumn)


def test_distinct_table(functional_alltypes):
    expr = functional_alltypes.distinct()
    assert isinstance(expr.op(), ops.Distinct)
    assert isinstance(expr, ir.Table)
    assert expr.op().table == functional_alltypes.op()


def test_nunique(functional_alltypes):
    expr = functional_alltypes.string_col.nunique()
    assert isinstance(expr.op(), ops.CountDistinct)


def test_isnull(table):
    expr = table["g"].isnull()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsNull)

    expr = ibis.literal("foo").isnull()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsNull)


def test_notnull(table):
    expr = table["g"].notnull()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.NotNull)

    expr = ibis.literal("foo").notnull()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.NotNull)


@pytest.mark.parametrize("column", ["e", "f"], ids=["float32", "double"])
def test_isnan_isinf_column(table, column):
    expr = table[column].isnan()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsNan)

    expr = table[column].isinf()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsInf)


@pytest.mark.parametrize("value", [1.3, np.nan, np.inf, -np.inf])
def test_isnan_isinf_scalar(value):
    expr = ibis.literal(value).isnan()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsNan)

    expr = ibis.literal(value).isinf()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsInf)


@pytest.mark.parametrize(
    ["column", "operation"],
    [
        ("d", "cumsum"),
        ("d", "cummean"),
        ("d", "cummin"),
        ("d", "cummax"),
        ("h", "cumany"),
        ("h", "cumall"),
    ],
)
def test_cumulative_yield_array_types(table, column, operation):
    expr = getattr(getattr(table, column), operation)()
    assert isinstance(expr, ir.Column)


@pytest.fixture(params=["ln", "log", "log2", "log10"])
def log(request):
    return operator.methodcaller(request.param)


@pytest.mark.parametrize("column", list("abcdef"))
def test_log(table, log, column):
    result = log(table[column])
    assert isinstance(result, ir.FloatingColumn)

    # is this what we want?
    # assert result.get_name() == c


def test_log_string(table):
    g = table.g

    with pytest.raises(ValidationError):
        ops.Log(g, None).to_expr()


@pytest.mark.parametrize("klass", [ops.Ln, ops.Log2, ops.Log10])
def test_log_variants_string(table, klass):
    g = table.g

    with pytest.raises(ValidationError):
        klass(g).to_expr()


def test_log_boolean(table, log):
    # boolean not implemented for these
    h = table["h"]
    with pytest.raises(ValidationError):
        log(h)


def test_log_literal(log):
    assert isinstance(log(ibis.literal(5)), ir.FloatingScalar)
    assert isinstance(log(ibis.literal(5.5)), ir.FloatingScalar)


def test_cast_same_type_noop(table):
    c = table.g
    assert c.cast("string") is c

    i = ibis.literal(5)
    assert i.cast("int8") is i


@pytest.mark.parametrize("type", ["int8", "int32", "double", "float32"])
def test_string_to_number(table, type):
    casted = table.g.cast(type)
    casted_literal = ibis.literal("5").cast(type).name("bar")

    assert isinstance(casted, ir.Column)
    assert casted.type() == dt.dtype(type)

    assert isinstance(casted_literal, ir.Scalar)
    assert casted_literal.type() == dt.dtype(type)
    assert casted_literal.get_name() == "bar"


@pytest.mark.parametrize("col", list("abcdefh"))
def test_number_to_string_column(table, col):
    casted = table[col].cast("string")
    assert isinstance(casted, ir.StringColumn)


def test_number_to_string_scalar():
    casted_literal = ibis.literal(5).cast("string").name("bar")
    assert isinstance(casted_literal, ir.StringScalar)
    assert casted_literal.get_name() == "bar"


def test_casted_exprs_are_named(table):
    expr = table.f.cast("string")
    assert expr.get_name() == "Cast(f, string)"

    # it works! per GH #396
    expr.value_counts()


@pytest.mark.parametrize("col", list("abcdef"))
def test_negate(table, col):
    c = table[col]
    result = -c
    assert isinstance(result, type(c))
    assert isinstance(result.op(), ops.Negate)


@pytest.mark.parametrize("op", [operator.neg, operator.invert])
@pytest.mark.parametrize("value", [True, False])
def test_negate_boolean_scalar(op, value):
    result = op(ibis.literal(value))
    assert isinstance(result, ir.BooleanScalar)
    assert isinstance(result.op(), ops.Not)


@pytest.mark.parametrize("op", [operator.neg, operator.invert])
def test_negate_boolean_column(table, op):
    result = op(table["h"])
    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(result.op(), ops.Not)


@pytest.mark.parametrize("column", ["a", "b", "c", "d", "e", "f", "g", "h"])
@pytest.mark.parametrize("how", ["first", "last", "heavy"])
@pytest.mark.parametrize("condition_fn", [lambda t: None, lambda t: t.a > 8])
def test_arbitrary(table, column, how, condition_fn):
    col = table[column]
    where = condition_fn(table)
    expr = col.arbitrary(how=how, where=where)
    assert expr.type() == col.type()
    assert isinstance(expr, ir.Scalar)
    assert isinstance(expr.op(), ops.Arbitrary)


@pytest.mark.parametrize(
    ["column", "operation"],
    [
        ("h", lambda column: column.any()),
        ("h", lambda column: column.notany()),
        ("h", lambda column: column.all()),
        ("c", lambda column: (column == 0).any()),
        ("c", lambda column: (column == 0).all()),
    ],
)
def test_any_all_notany(table, column, operation):
    expr = operation(table[column])
    assert isinstance(expr, ir.BooleanScalar)
    assert expr.op().find(ops.Reduction)


@pytest.mark.parametrize(
    "operation",
    [
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ],
)
@pytest.mark.parametrize("column", list("abcdef"))
@pytest.mark.parametrize("case", [2, 2**9, 2**17, 2**33, 1.5])
def test_numbers_compare_numeric_literal(table, operation, column, case):
    ex_op_class = {
        operator.eq: ops.Equals,
        operator.ne: ops.NotEquals,
        operator.le: ops.LessEqual,
        operator.lt: ops.Less,
        operator.ge: ops.GreaterEqual,
        operator.gt: ops.Greater,
    }

    col = table[column]

    result = operation(col, case)
    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(result.op(), ex_op_class[operation])


def test_boolean_comparisons(table):
    bool_col = table.h

    result = bool_col == True  # noqa: E712
    assert isinstance(result, ir.BooleanColumn)

    result = bool_col == False  # noqa: E712
    assert isinstance(result, ir.BooleanColumn)


@pytest.mark.parametrize(
    "operation",
    [
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ],
)
def test_string_comparisons(table, operation):
    string_col = table.g
    result = operation(string_col, "foo")
    assert isinstance(result, ir.BooleanColumn)


@pytest.mark.parametrize("operation", [operator.xor, operator.or_, operator.and_])
def test_boolean_logical_ops(table, operation):
    expr = table.a > 0

    result = operation(expr, table.h)
    assert isinstance(result, ir.BooleanColumn)

    result = operation(expr, True)
    refl_result = operation(True, expr)
    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(refl_result, ir.BooleanColumn)

    true = ibis.literal(True)
    false = ibis.literal(False)

    result = operation(true, false)
    assert isinstance(result, ir.BooleanScalar)


def test_and_(table):
    p1 = table.a > 1
    p2 = table.b > 1
    p3 = table.c > 1
    assert ibis.and_().equals(ibis.literal(True))
    assert ibis.and_(p1).equals(p1)
    assert ibis.and_(p1, p2).equals(p1 & p2)
    assert ibis.and_(p1, p2, p3).equals(p1 & p2 & p3)


def test_or_(table):
    p1 = table.a > 1
    p2 = table.b > 1
    p3 = table.c > 1
    assert ibis.or_().equals(ibis.literal(False))
    assert ibis.or_(p1).equals(p1)
    assert ibis.or_(p1, p2).equals(p1 | p2)
    assert ibis.or_(p1, p2, p3).equals(p1 | p2 | p3)


def test_null_column():
    t = ibis.table([("a", "string")], name="t")
    s = t.mutate(b=ibis.NA)
    assert s.b.type() == dt.null
    assert isinstance(s.b, ir.NullColumn)


def test_null_column_union():
    s = ibis.table([("a", "string"), ("b", "double")])
    t = ibis.table([("a", "string")])
    with pytest.raises(ibis.common.exceptions.RelationError):
        s.union(t.mutate(b=ibis.NA))  # needs a type
    assert s.union(t.mutate(b=ibis.NA.cast("double"))).schema() == s.schema()


def test_string_compare_numeric_array(table):
    with pytest.raises(com.IbisTypeError):
        table.g == table.f  # noqa: B015

    with pytest.raises(com.IbisTypeError):
        table.g == table.c  # noqa: B015


def test_string_compare_numeric_literal(table):
    with pytest.raises(com.IbisTypeError):
        table.g == ibis.literal(1.5)  # noqa: B015

    with pytest.raises(com.IbisTypeError):
        table.g == ibis.literal(5)  # noqa: B015


def test_between(table):
    result = table.f.between(0, 1)

    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(result.op(), ops.Between)

    # it works!
    result = table.g.between("a", "f")
    assert isinstance(result, ir.BooleanColumn)

    result = ibis.literal(1).between(table.a, table.c)
    assert isinstance(result, ir.BooleanColumn)

    result = ibis.literal(7).between(5, 10)
    assert isinstance(result, ir.BooleanScalar)

    # Cases where between should immediately fail, e.g. incomparables
    with pytest.raises(ValidationError):
        table.f.between("0", "1")

    with pytest.raises(ValidationError):
        table.f.between(0, "1")

    with pytest.raises(ValidationError):
        table.f.between("0", 1)


def test_chained_comparisons_not_allowed(table):
    with pytest.raises(ValueError):
        0 < table.f < 1  # noqa: B015


@pytest.mark.parametrize(
    "operation",
    [operator.add, operator.sub, operator.truediv],
)
@pytest.mark.parametrize(("left", "right"), [("d", "g"), ("g", "d")])
def test_binop_string_type_error(table, operation, left, right):
    a = table[left]
    b = table[right]

    with pytest.raises((TypeError, ValidationError)):
        operation(a, b)


@pytest.mark.parametrize(("left", "right"), [("d", "g"), ("g", "d")])
def test_string_mul(table, left, right):
    a = table[left]
    b = table[right]

    expr = a * b
    assert isinstance(expr, ir.StringColumn)
    assert isinstance(expr.op(), ops.Repeat)


@pytest.mark.parametrize(
    ["op", "name", "case", "ex_type"],
    [
        (operator.add, "a", 0, "int8"),
        (operator.add, "a", 5, "int16"),
        (operator.add, "a", 100000, "int32"),
        (operator.add, "a", -100000, "int32"),
        (operator.add, "a", 1.5, "double"),
        (operator.add, "b", 0, "int16"),
        (operator.add, "b", 5, "int32"),
        (operator.add, "b", -5, "int32"),
        (operator.add, "c", 0, "int32"),
        (operator.add, "c", 5, "int64"),
        (operator.add, "c", -5, "int64"),
        # technically this can overflow, but we allow it
        (operator.add, "d", 5, "int64"),
        (operator.mul, "a", 0, "int8"),
        (operator.mul, "a", 5, "int16"),
        (operator.mul, "a", 2**24, "int32"),
        (operator.mul, "a", -(2**24) + 1, "int32"),
        (operator.mul, "a", 1.5, "double"),
        (operator.mul, "b", 0, "int16"),
        (operator.mul, "b", 5, "int32"),
        (operator.mul, "b", -5, "int32"),
        (operator.mul, "c", 0, "int32"),
        (operator.mul, "c", 5, "int64"),
        (operator.mul, "c", -5, "int64"),
        # technically this can overflow, but we allow it
        (operator.mul, "d", 5, "int64"),
        (operator.sub, "a", 5, "int16"),
        (operator.sub, "a", 100000, "int32"),
        (operator.sub, "a", -100000, "int32"),
        (operator.sub, "a", 1.5, "double"),
        (operator.sub, "b", 5, "int32"),
        (operator.sub, "b", -5, "int32"),
        (operator.sub, "c", 5, "int64"),
        (operator.sub, "c", -5, "int64"),
        # technically this can overflow, but we allow it
        (operator.sub, "d", 5, "int64"),
        (operator.truediv, "a", 5, "double"),
        (operator.truediv, "a", 1.5, "double"),
        (operator.truediv, "b", 5, "double"),
        (operator.truediv, "b", -5, "double"),
        (operator.truediv, "c", 5, "double"),
        (operator.pow, "a", 0, "double"),
        (operator.pow, "b", 0, "double"),
        (operator.pow, "c", 0, "double"),
        (operator.pow, "d", 0, "double"),
        (operator.pow, "e", 0, "float32"),
        (operator.pow, "f", 0, "double"),
        (operator.pow, "a", 2, "double"),
        (operator.pow, "b", 2, "double"),
        (operator.pow, "c", 2, "double"),
        (operator.pow, "d", 2, "double"),
        (operator.pow, "a", 1.5, "double"),
        (operator.pow, "b", 1.5, "double"),
        (operator.pow, "c", 1.5, "double"),
        (operator.pow, "d", 1.5, "double"),
        (operator.pow, "e", 2, "float32"),
        (operator.pow, "f", 2, "double"),
        (operator.pow, "a", -2, "double"),
        (operator.pow, "b", -2, "double"),
        (operator.pow, "c", -2, "double"),
        (operator.pow, "d", -2, "double"),
    ],
    ids=lambda arg: str(getattr(arg, "__name__", arg)),
)
def test_literal_promotions(table, op, name, case, ex_type):
    col = table[name]

    result = op(col, case)
    assert result.type() == dt.dtype(ex_type)

    result = op(case, col)
    assert result.type() == dt.dtype(ex_type)


@pytest.mark.parametrize(
    ("op", "left_fn", "right_fn", "ex_type"),
    [
        (operator.sub, lambda t: t["a"], lambda t: 0, "int8"),
        (operator.sub, lambda t: 0, lambda t: t["a"], "int16"),
        (operator.sub, lambda t: t["b"], lambda t: 0, "int16"),
        (operator.sub, lambda t: 0, lambda t: t["b"], "int32"),
        (operator.sub, lambda t: t["c"], lambda t: 0, "int32"),
        (operator.sub, lambda t: 0, lambda t: t["c"], "int64"),
    ],
    ids=lambda arg: str(getattr(arg, "__name__", arg)),
)
def test_zero_subtract_literal_promotions(table, op, left_fn, right_fn, ex_type):
    # in case of zero subtract the order of operands matters
    left, right = left_fn(table), right_fn(table)
    result = op(left, right)

    assert result.type() == dt.dtype(ex_type)


def test_substitute_dict():
    table = ibis.table([("foo", "string"), ("bar", "string")], "t1")
    subs = {"a": "one", "b": table.bar}

    result = table.foo.substitute(subs)
    expected = (
        table.foo.case().when("a", "one").when("b", table.bar).else_(table.foo).end()
    )
    assert_equal(result, expected)

    result = table.foo.substitute(subs, else_=ibis.NA)
    expected = (
        table.foo.case().when("a", "one").when("b", table.bar).else_(ibis.NA).end()
    )
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "typ",
    [
        "array<map<string, array<array<double>>>>",
        "string",
        "double",
        "float32",
    ],
)
def test_not_without_boolean(typ):
    t = ibis.table([("a", typ)], name="t")
    c = t.a
    with pytest.raises(TypeError):
        ~c  # noqa: B018


@pytest.mark.parametrize(
    ("fn", "expected_op"),
    [
        param(lambda t: t.int_col & t.smallint_col, ops.BitwiseAnd, id="and"),
        param(lambda t: t.int_col | t.smallint_col, ops.BitwiseOr, id="or"),
        param(lambda t: t.int_col ^ t.smallint_col, ops.BitwiseXor, id="xor"),
        param(
            lambda t: t.int_col << t.smallint_col,
            ops.BitwiseLeftShift,
            id="lshift",
        ),
        param(
            lambda t: t.int_col >> t.smallint_col,
            ops.BitwiseRightShift,
            id="rshift",
        ),
        param(lambda t: operator.inv(t.int_col), ops.BitwiseNot, id="not"),
    ],
)
def test_bitwise_exprs(fn, expected_op):
    t = ibis.table(dict(int_col="int32", smallint_col="int16"), name="t")
    expr = fn(t)
    assert isinstance(expr.op(), expected_op)


@pytest.mark.parametrize(
    ("position", "names"),
    [
        (0, "foo"),
        (1, "bar"),
        ([0], ["foo"]),
        ([1], ["bar"]),
        ([0, 1], ["foo", "bar"]),
        ([1, 0], ["bar", "foo"]),
    ],
)
@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t, args: t[args],
        lambda t, args: t.order_by(args),
        lambda t, args: t.group_by(args).aggregate(bar_avg=t.bar.mean()),
    ],
)
def test_table_operations_with_integer_column(position, names, expr_func):
    t = ibis.table([("foo", "string"), ("bar", "double")])
    result = expr_func(t, position)
    expected = expr_func(t, names)
    assert result.equals(expected)


@pytest.mark.parametrize("value", ["abcdefg", ["a", "b", "c"], [1, 2, 3]])
@pytest.mark.parametrize("operation", ["pow", "sub", "truediv", "floordiv", "mod"])
def test_generic_value_api_no_arithmetic(value, operation):
    func = getattr(operator, operation)
    expr = ibis.literal(value)
    with pytest.raises(TypeError):
        func(expr, expr)


@pytest.mark.parametrize(
    ("value", "expected"), [(5, dt.int8), (5.4, dt.double), ("abc", dt.string)]
)
def test_fillna_null(value, expected):
    assert ibis.NA.fillna(value).type().equals(expected)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (literal("2017-04-01"), date(2017, 4, 2)),
        (date(2017, 4, 2), literal("2017-04-01")),
        (literal("2017-04-01 01:02:33"), datetime(2017, 4, 1, 1, 3, 34)),
        (datetime(2017, 4, 1, 1, 3, 34), literal("2017-04-01 01:02:33")),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        lambda left, right: ibis.timestamp("2017-04-01 00:02:34").between(left, right),
        lambda left, right: ibis.timestamp("2017-04-01")
        .cast(dt.date)
        .between(left, right),
    ],
)
def test_string_temporal_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ("value", "type", "expected_type_class"),
    [
        (2.21, "decimal", dt.Decimal),
        (3.14, "float64", dt.Float64),
        (4.2, "int64", dt.Float64),
        (4, "int64", dt.Int64),
    ],
)
def test_decimal_modulo_output_type(value, type, expected_type_class):
    t = ibis.table([("a", type)])
    expr = t.a % value
    assert isinstance(expr.type(), expected_type_class)


@pytest.mark.parametrize(
    ("left", "right"),
    [(literal("10:00"), time(10, 0)), (time(10, 0), literal("10:00"))],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
def test_time_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (literal("10:00"), date(2017, 4, 2)),
        (literal("10:00"), datetime(2017, 4, 2, 1, 1)),
        (literal("10:00"), literal("2017-04-01")),
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_time_timestamp_invalid_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            # same value type, same name
            ibis.param(dt.timestamp),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # different value type, same name
            ibis.param(dt.date),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # same value type, different name
            ibis.param(dt.timestamp),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # different value type, different name
            ibis.param(dt.date),
            ibis.param(dt.timestamp),
            False,
        ),
    ],
)
def test_scalar_parameter_compare(left, right, expected):
    assert left.equals(right) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # different Python class, left side is param
        param(ibis.param(dt.timestamp), dt.date, id="param_dtype"),
        # different Python class, right side is param
        param(dt.date, ibis.param(dt.timestamp), id="dtype_param"),
    ],
)
def test_scalar_parameter_invalid_compare(left, right):
    with pytest.raises(TypeError):
        left.equals(right)


@pytest.mark.parametrize(
    ("case", "creator"),
    [
        (datetime.now(), toolz.compose(methodcaller("time"), ibis.timestamp)),
        ("now", toolz.compose(methodcaller("time"), ibis.timestamp)),
        (datetime.now().time(), ibis.time),
        ("10:37", ibis.time),
    ],
)
@pytest.mark.parametrize(
    ("left", "right"), [(1, "a"), ("a", 1), (1.0, 2.0), (["a"], [1])]
)
def test_between_time_failure_time(case, creator, left, right):
    value = creator(case)
    with pytest.raises(ValidationError):
        value.between(left, right)


def test_empty_array_as_argument():
    class Foo(ir.Expr):
        pass

    class FooNode(ops.Node):
        value: ops.Value[dt.Array[dt.Int64], ds.Any]

        def to_expr(self):
            return Foo(self)

    node = FooNode([])
    assert node.value.value == ()
    assert node.value.dtype == dt.Array(dt.Int64)


def test_nullable_column_propagated():
    t = ibis.table(
        [
            ("a", dt.Int32(nullable=True)),
            ("b", dt.Int32(nullable=False)),
            ("c", dt.String(nullable=False)),
            ("d", dt.float64),  # nullable by default
            ("f", dt.Float64(nullable=False)),
        ]
    )

    assert t.a.type().nullable is True
    assert t.b.type().nullable is False
    assert t.c.type().nullable is False
    assert t.d.type().nullable is True
    assert t.f.type().nullable is False

    s = t.a + t.d
    assert s.type().nullable is True

    s = t.b + t.d
    assert s.type().nullable is True

    s = t.b + t.f
    assert s.type().nullable is False


@pytest.mark.parametrize(
    "base_expr",
    [
        ibis.table([("interval_col", dt.Interval(unit="D"))]).interval_col,
        ibis.interval(seconds=42),
    ],
)
def test_interval_negate(base_expr):
    expr = -base_expr
    expr2 = base_expr.negate()
    assert isinstance(expr.op(), ops.Negate)
    assert expr.equals(expr2)


def test_large_timestamp():
    expr = ibis.timestamp("4567-02-03")
    expected = datetime(year=4567, month=2, day=3)
    result = expr.op().value
    assert result == expected


def test_timestamp_with_timezone():
    expr = ibis.timestamp("2017-01-01", timezone=None)
    expected = datetime(2017, 1, 1, tzinfo=None)
    assert expr.op().value == expected

    expr = ibis.timestamp("2017-01-01", timezone="UTC")
    expected = datetime(2017, 1, 1, tzinfo=pytz.timezone("UTC"))
    assert expr.op().value == expected


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_timestamp_timezone_type(tz):
    expr = ibis.timestamp("2017-01-01", timezone=tz)
    expected = dt.Timestamp(timezone=tz)
    assert expected == expr.op().dtype


def test_map_get_broadcast():
    t = ibis.table([("a", "string")], name="t")
    lookup_table = ibis.literal({"a": 1, "b": 2})

    expr = lookup_table.get(t.a)
    assert isinstance(expr, ir.IntegerColumn)


def test_map_getitem_broadcast():
    t = ibis.table([("a", "string")], name="t")
    lookup_table = ibis.literal({"a": 1, "b": 2})
    expr = lookup_table[t.a]
    assert isinstance(expr, ir.IntegerColumn)


def test_map_keys_output_type():
    mapping = ibis.literal({"a": 1, "b": 2})
    assert mapping.keys().type() == dt.Array(dt.string)


def test_map_values_output_type():
    mapping = ibis.literal({"a": 1, "b": 2})
    assert mapping.values().type() == dt.Array(dt.int8)


def test_scalar_isin_map_keys():
    mapping = ibis.literal({"a": 1, "b": 2})
    key = ibis.literal("a")
    expr = key.isin(mapping.keys())
    assert isinstance(expr, ir.BooleanScalar)


def test_column_isin_array():
    # scalar case
    t = ibis.table([("a", "string")], name="t")
    expr = t.a.isin(ibis.array(["a", "b"]))
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.ArrayContains)
    assert expr.op().shape.is_columnar()

    # columnar case
    t = ibis.table([("a", "string"), ("b", "array<string>")], name="t")
    expr = t.a.isin(t.b)
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.ArrayContains)
    assert expr.op().shape.is_columnar()


def test_column_isin_map_keys():
    t = ibis.table([("a", "string")], name="t")
    mapping = ibis.literal({"a": 1, "b": 2})
    expr = t.a.isin(mapping.keys())
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.ArrayContains)


def test_map_get_with_compatible_value_smaller():
    value = ibis.literal({"A": 1000, "B": 2000})
    expr = value.get("C", 3)
    assert value.type() == dt.Map(dt.string, dt.int16)
    assert expr.type() == dt.int16


def test_map_get_with_compatible_value_bigger():
    value = ibis.literal({"A": 1, "B": 2})
    expr = value.get("C", 3000)
    assert value.type() == dt.Map(dt.string, dt.int8)
    assert expr.type() == dt.int16


def test_map_get_with_incompatible_value_different_kind():
    value = ibis.literal({"A": 1000, "B": 2000})
    assert value.get("C", 3.0).type() == dt.float64


@pytest.mark.parametrize("null_value", [None, ibis.NA])
def test_map_get_with_null_on_not_nullable(null_value):
    map_type = dt.Map(dt.string, dt.Int16(nullable=False))
    value = ibis.literal({"A": 1000, "B": 2000}).cast(map_type)
    assert value.type() == map_type
    expr = value.get("C", null_value)
    assert expr.type() == dt.Int16(nullable=True)


@pytest.mark.parametrize("null_value", [None, ibis.NA])
def test_map_get_with_null_on_nullable(null_value):
    value = ibis.literal({"A": 1000, "B": None})
    result = value.get("C", null_value)
    assert result.type().nullable


@pytest.mark.parametrize("null_value", [None, ibis.NA])
def test_map_get_with_null_on_null_type_with_null(null_value):
    value = ibis.literal({"A": None, "B": None})
    result = value.get("C", null_value)
    assert result.type().nullable


def test_map_get_with_null_on_null_type_with_non_null():
    value = ibis.literal({"A": None, "B": None})
    assert value.get("C", 1).type() == dt.int8


def test_map_get_with_incompatible_value():
    value = ibis.literal({"A": 1000, "B": 2000})
    with pytest.raises(TypeError):
        value.get("C", ["A"])


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (datetime.now(), dt.timestamp),
        (datetime.now().date(), dt.date),
        (datetime.now().time(), dt.time),
    ],
)
def test_invalid_negate(value, expected_type):
    expr = ibis.literal(value)
    assert expr.type() == expected_type
    with pytest.raises(TypeError):
        -expr  # noqa: B018


@pytest.mark.parametrize(
    "type",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int16,
        np.int32,
        np.int64,
        np.int64,
        np.int8,
        np.timedelta64,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uint64,
        np.uint8,
        float,
        int,
    ],
)
def test_valid_negate(type):
    expr = ibis.literal(1)
    assert -expr is not None


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ibis.literal(1), ibis.literal(1.0), dt.float64),
        (ibis.literal("a"), ibis.literal("b"), dt.string),
        (ibis.literal(1.0), ibis.literal(1), dt.float64),
        (ibis.literal(1), ibis.literal(1), dt.int8),
        (ibis.literal(1), ibis.literal(1000), dt.int16),
        (ibis.literal(2**16), ibis.literal(2**17), dt.int32),
        (ibis.literal(2**50), ibis.literal(1000), dt.int64),
        (ibis.literal([1, 2]), ibis.literal([1, 2]), dt.Array(dt.int8)),
        (ibis.literal(["a"]), ibis.literal([]), dt.Array(dt.string)),
        (ibis.literal([]), ibis.literal(["a"]), dt.Array(dt.string)),
        (ibis.literal([]), ibis.literal([]), dt.Array(dt.null)),
    ],
)
def test_nullif_type(left, right, expected):
    assert left.nullif(right).type() == expected


@pytest.mark.parametrize(("left", "right"), [(ibis.literal(1), ibis.literal("a"))])
def test_nullif_fail(left, right):
    with pytest.raises(com.IbisTypeError):
        left.nullif(right)
    with pytest.raises(com.IbisTypeError):
        right.nullif(left)


@pytest.mark.parametrize(
    "join_method",
    [
        "left_join",
        "right_join",
        "inner_join",
        "outer_join",
        "asof_join",
    ],
)
def test_select_on_unambiguous_join(join_method):
    t = ibis.table([("a0", dt.int64), ("b1", dt.string)], name="t")
    s = ibis.table([("a1", dt.int64), ("b2", dt.string)], name="s")
    method = getattr(t, join_method)
    join = method(s, t.b1 == s.b2)
    expr1 = join["a0", "a1"]
    expr2 = join[["a0", "a1"]]
    expr3 = join.select(["a0", "a1"])
    assert expr1.equals(expr2)
    assert expr1.equals(expr3)


def test_chained_select_on_join():
    t = ibis.table([("a", dt.int64)], name="t")
    s = ibis.table([("a", dt.int64), ("b", dt.string)], name="s")
    join = t.join(s)[t.a, s.b]
    expr1 = join["a", "b"]
    expr2 = join.select(["a", "b"])
    assert expr1.equals(expr2)


def test_repr_list_of_lists():
    lit = ibis.literal([[1]])
    repr(lit)


def test_repr_list_of_lists_in_table():
    t = ibis.table([("a", "int64")], name="t")
    lit = ibis.literal([[1]])
    expr = t[t, lit.name("array_of_array")]
    repr(expr)


@pytest.mark.parametrize(
    ("expr", "expected_type"),
    [
        (ibis.coalesce(ibis.NA, 1), dt.int8),
        (ibis.coalesce(1, ibis.NA), dt.int8),
        (ibis.coalesce(ibis.NA, 1000), dt.int16),
        (ibis.coalesce(ibis.NA), dt.null),
        (ibis.coalesce(ibis.NA, ibis.NA), dt.null),
        (
            ibis.coalesce(ibis.NA, ibis.NA.cast("array<string>")),
            dt.Array(dt.string),
        ),
    ],
)
def test_coalesce_type_inference_with_nulls(expr, expected_type):
    assert expr.type() == expected_type


def test_literal_hash():
    expr = ibis.literal(1)
    op = expr.op()
    # hashing triggers computation and storage of the op's hash
    result1 = hash(op)
    assert op.__precomputed_hash__ == result1
    assert hash(op) == result1


@pytest.mark.parametrize(
    "op_name",
    [
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "pow",
        "mod",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ],
)
@pytest.mark.parametrize(
    ("left", "right"),
    [
        param(_.a, 2, id="a_int"),
        param(2, _.a, id="int_a"),
        param(_.a, _.b, id="a_b"),
    ],
)
def test_deferred(op_name, left, right):
    t = ibis.table(dict(a="int64"), name="t")

    op = getattr(operator, op_name)
    expr = (
        t.mutate(b=_.a)
        .mutate(expr=op(left, right))
        .group_by(key=_.b.cast("string"))
        .aggregate(avg=_.b.mean(where=_.a < 1), neg_sum=-_.b.sum())
        .mutate(c=_.avg <= 1.0, d=~(_["neg_sum"] > 2))
        .filter(_.c)
    )

    assert expr.schema() == ibis.schema(
        dict(
            key="string",
            avg="float64",
            neg_sum="int64",
            c="bool",
            d="bool",
        )
    )


@pytest.mark.parametrize("bool_op_name", ["or_", "and_", "xor"])
@pytest.mark.parametrize(
    ("bool_left", "bool_right"),
    [
        param(_.a < 1, True, id="a_true"),
        param(False, _.a > 1, id="false_a"),
        param(_.a == 2, _.b == 1, id="expr_expr"),
    ],
)
def test_deferred_bool(bool_op_name, bool_left, bool_right):
    t = ibis.table(dict(a="int64"), name="t")

    bool_op = getattr(operator, bool_op_name)
    expr = t.mutate(b=_.a).mutate(c=bool_op(bool_left, bool_right)).filter(_.c)

    assert expr.schema() == ibis.schema(dict(a="int64", b="int64", c="bool"))


@pytest.mark.parametrize(
    ("op_name", "expected_left", "expected_right"),
    [
        param("add", attrgetter("a"), lambda _: ibis.literal(2), id="add"),
        param("sub", lambda _: ibis.literal(2), attrgetter("a"), id="sub"),
        param("mul", attrgetter("a"), lambda _: ibis.literal(2), id="mul"),
        param(
            "truediv",
            lambda _: ibis.literal(2),
            attrgetter("a"),
            id="truediv",
        ),
        param(
            "floordiv",
            lambda _: ibis.literal(2),
            attrgetter("a"),
            id="floordiv",
        ),
        param("pow", lambda _: ibis.literal(2), attrgetter("a"), id="pow"),
        param("mod", lambda _: ibis.literal(2), attrgetter("a"), id="mod"),
    ],
)
def test_deferred_r_ops(op_name, expected_left, expected_right):
    t = ibis.table(dict(a="int64"), name="t")

    left = 2
    right = _.a

    op = getattr(operator, op_name)
    expr = t[op(left, right).name("b")]

    op = expr.op().selections[0].arg
    assert op.left.equals(expected_left(t).op())
    assert op.right.equals(expected_right(t).op())


@pytest.mark.parametrize(
    ("expr_fn", "expected_type"),
    [
        (lambda t: ibis.ifelse(t.a == 1, t.b, ibis.NA), dt.string),
        (lambda t: ibis.ifelse(t.a == 1, t.b, t.a.cast("string")), dt.string),
        (
            lambda t: ibis.ifelse(t.a == 1, t.b, t.a.cast("!string")),
            dt.string.copy(nullable=False),
        ),
        (lambda _: ibis.ifelse(True, ibis.NA, ibis.NA), dt.null),
        (lambda _: ibis.ifelse(False, ibis.NA, ibis.NA), dt.null),
    ],
)
def test_non_null_with_null_precedence(expr_fn, expected_type):
    t = ibis.table(dict(a="int64", b="!string"), name="t")
    expr = expr_fn(t)
    assert expr.type() == expected_type


def test_struct_names_types_fields():
    s = ibis.struct(dict(a=1, b="2", c=[[1.0], [], [None]]))
    assert s.names == ("a", "b", "c")
    assert s.types == (dt.int8, dt.string, dt.Array(dt.Array(dt.float64)))
    assert s.fields == dict(a=dt.int8, b=dt.string, c=dt.Array(dt.Array(dt.float64)))
    assert isinstance(s.fields, frozendict)


@pytest.mark.parametrize(
    ["arg", "typestr", "type"],
    [
        ([1, 2, 3], None, dt.Array(dt.int8)),
        ([1, 2, 3], "array<int16>", dt.Array(dt.int16)),
        ([1, 2, 3.0], None, dt.Array(dt.double)),
        (["a", "b", "c"], None, dt.Array(dt.string)),
    ],
)
def test_array_literal(arg, typestr, type):
    x = ibis.literal(arg, type=typestr)
    assert x.op().value == tuple(arg)
    assert x.type() == type


def test_array_length_scalar():
    raw_value = [1, 2, 4]
    value = ibis.literal(raw_value)
    expr = value.length()
    assert isinstance(expr.op(), ops.ArrayLength)


def double_int(x):
    return x * 2


def double_float(x):
    return x * 2.0


def is_negative(x):
    return x < 0


def test_array_map():
    arr = ibis.array([1, 2, 3])

    result_int = arr.map(double_int)
    result_float = arr.map(double_float)

    assert result_int.type() == dt.Array(dt.int16)
    assert result_float.type() == dt.Array(dt.float64)


def test_array_map_partial():
    arr = ibis.array([1, 2, 3])

    def add(x, y):
        return x + y

    result = arr.map(functools.partial(add, y=2))
    assert result.type() == dt.Array(dt.int16)


def test_array_filter():
    arr = ibis.array([1, 2, 3])
    result = arr.filter(is_negative)
    assert result.type() == arr.type()


def test_array_filter_partial():
    arr = ibis.array([1, 2, 3])

    def equal(x, y):
        return x == y

    result = arr.filter(functools.partial(equal, y=2))
    assert result.type() == arr.type()


@pytest.mark.parametrize(
    ("func", "expected_type"),
    [
        param(ibis.timestamp, dt.timestamp, id="timestamp"),
        param(ibis.date, dt.date, id="date"),
        param(ibis.time, dt.time, id="time"),
        param(ibis.coalesce, dt.timestamp, id="coalesce"),
        param(ibis.greatest, dt.timestamp, id="greatest"),
        param(ibis.least, dt.timestamp, id="least"),
        param(
            lambda ts: ibis.ifelse(ts.notnull(), ts, ts - ibis.interval(days=1)),
            dt.timestamp,
            id="ifelse",
        ),
    ],
)
def test_deferred_function_call(func, expected_type):
    t = ibis.table(dict(ts="timestamp"), name="t")
    expr = t.select(col=func(_.ts))
    assert expr["col"].type() == expected_type


@pytest.mark.parametrize(
    "case",
    [
        param(lambda: (ibis.array([1, _]), ibis.array([1, 2])), id="array"),
        param(
            lambda: (ibis.map({"x": 1, "y": _}), ibis.map({"x": 1, "y": 2})), id="map"
        ),
        param(
            lambda: (ibis.struct({"x": 1, "y": _}), ibis.struct({"x": 1, "y": 2})),
            id="struct",
        ),
    ],
)
def test_deferred_nested_types(case):
    expr, sol = case()
    assert expr.resolve(2).equals(sol)


def test_numpy_ufuncs_dont_cast_columns():
    t = ibis.table(dict.fromkeys("abcd", "int"))

    # Adding a numpy array doesn't implicitly compute
    arr = np.array([1, 2, 3])
    for left, right in [(t.a, arr), (arr, t.a)]:
        with pytest.raises(TypeError):
            left + right

    # Adding a numpy scalar works and results in a new expr
    x = np.int64(1)
    for expr in [t.a + x, x + t.a]:
        assert expr.equals(t.a + ibis.literal(x))


@pytest.mark.parametrize(
    "operation",
    [operator.lt, operator.gt, operator.ge, operator.le, operator.eq, operator.ne],
)
def test_logical_comparison_rlz_incompatible_error(table, operation):
    with pytest.raises(com.IbisTypeError, match=r"b:int16 and Literal\(foo\):string"):
        operation(table.b, "foo")


def test_case_rlz_incompatible_error(table):
    with pytest.raises(com.IbisTypeError, match=r"a:int8 and Literal\(foo\):string"):
        table.a == "foo"  # noqa: B015


@pytest.mark.parametrize("func", [ibis.asc, ibis.desc])
def test_group_by_order_by_deferred(func):
    from ibis import _

    table = ibis.table(dict(x="string", y="int"), name="t")
    expr = table.group_by(_.x).aggregate(mean_y=_.y.mean()).order_by(func(_.mean_y))
    assert isinstance(expr, ir.Table)


def test_rowid_only_physical_tables():
    table = ibis.table({"x": "int", "y": "string"}, name="t")

    table.rowid()  # works
    table[table.rowid(), table.x].filter(_.x > 10)  # works
    with pytest.raises(com.IbisTypeError, match="only valid for physical tables"):
        table.filter(table.x > 0).rowid()


def test_where_shape():
    # GH-5191
    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = ibis.literal(True).ifelse(t.a, -t.a)
    assert isinstance(expr, ir.IntegerColumn)
    assert isinstance(expr, ir.ColumnExpr)


def test_quantile_shape():
    t = ibis.table([("a", "float64")])

    b1 = t.a.quantile(0.25).name("br2")
    assert isinstance(b1, ir.Scalar)

    projs = [b1]
    expr = t.select(projs)
    (b1,) = expr.op().selections

    assert b1.shape.is_columnar()


def test_sample():
    t = ibis.table({"x": "int64", "y": "string"})

    expr = t.sample(1)
    assert expr.equals(t)

    expr = t.sample(0)
    assert expr.equals(t.limit(0))

    expr = t.sample(0.5, method="block", seed=1234)
    assert expr.schema() == t.schema()
    op = expr.op()
    assert op.fraction == 0.5
    assert op.method == "block"
    assert op.seed == 1234
