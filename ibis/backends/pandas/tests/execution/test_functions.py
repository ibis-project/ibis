from __future__ import annotations

import decimal
import functools
import math
import operator
from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.pandas.execution import execute
from ibis.backends.pandas.tests.conftest import TestConf as tm
from ibis.backends.pandas.udf import udf


@pytest.mark.parametrize(
    "op",
    [
        # comparison
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
def test_binary_operations(t, df, op):
    expr = op(t.plain_float64, t.plain_int64)
    result = expr.execute()
    expected = op(df.plain_float64, df.plain_int64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.execute()
    expected = op(df.plain_int64 == 1, df.plain_int64 == 2)
    tm.assert_series_equal(result, expected)


def operate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except decimal.InvalidOperation:
            return decimal.Decimal("NaN")

    return wrapper


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        param(methodcaller("round"), round, id="round"),
        param(
            methodcaller("round", 2),
            lambda x: x.quantize(decimal.Decimal(".00")),
            id="round_2",
        ),
        param(
            methodcaller("round", 0),
            lambda x: x.quantize(decimal.Decimal("0.")),
            id="round_0",
        ),
        param(methodcaller("ceil"), lambda x: decimal.Decimal(math.ceil(x)), id="ceil"),
        param(
            methodcaller("floor"), lambda x: decimal.Decimal(math.floor(x)), id="floor"
        ),
        param(methodcaller("exp"), methodcaller("exp"), id="exp"),
        param(
            methodcaller("sign"),
            lambda x: x if not x else decimal.Decimal(1).copy_sign(x),
            id="sign",
        ),
        param(methodcaller("sqrt"), operate(lambda x: x.sqrt()), id="sqrt"),
        param(
            methodcaller("log", 2),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln()),
            id="log_2",
        ),
        param(methodcaller("ln"), operate(lambda x: x.ln()), id="ln"),
        param(
            methodcaller("log2"),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln()),
            id="log2",
        ),
        param(methodcaller("log10"), operate(lambda x: x.log10()), id="log10"),
    ],
)
def test_math_functions_decimal(t, df, ibis_func, pandas_func):
    dtype = dt.Decimal(12, 3)
    expr = ibis_func(t.float64_as_strings.cast(dtype))
    result = expr.execute()
    context = decimal.Context(prec=dtype.precision)
    expected = df.float64_as_strings.apply(
        lambda x: context.create_decimal(x).quantize(
            decimal.Decimal(
                f"{'0' * (dtype.precision - dtype.scale)}.{'0' * dtype.scale}"
            )
        )
    ).apply(pandas_func)

    result[result.apply(math.isnan)] = -99999
    expected[expected.apply(math.isnan)] = -99999
    tm.assert_series_equal(result, expected.astype(expr.type().to_pandas()))


def test_round_decimal_with_negative_places(t):
    type = dt.Decimal(12, 3)
    expr = t.float64_as_strings.cast(type).round(-1)
    result = expr.execute()
    expected = pd.Series(
        list(map(decimal.Decimal, ["1.0E+2", "2.3E+2", "-1.00E+3"])),
        name="float64_as_strings",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        (lambda x: x.quantile(0), lambda x: x.quantile(0)),
        (lambda x: x.quantile(1), lambda x: x.quantile(1)),
        (lambda x: x.quantile(0.5), lambda x: x.quantile(0.5)),
    ],
)
def test_quantile(t, df, ibis_func, pandas_func):
    result = ibis_func(t.float64_with_zeros).execute()
    expected = pandas_func(df.float64_with_zeros)
    assert result == expected

    assert result == expected

    result = ibis_func(t.int64_with_zeros).execute()
    expected = pandas_func(df.int64_with_zeros)
    assert result == expected


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        (
            lambda x: x.quantile([0.25, 0.75]),
            lambda x: np.array(x.quantile([0.25, 0.75])),
        )
    ],
)
@pytest.mark.parametrize("column", ["float64_with_zeros", "int64_with_zeros"])
def test_quantile_multi(t, df, ibis_func, pandas_func, column):
    expr = ibis_func(t[column])
    result = expr.execute()
    expected = pandas_func(df[column])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("ibis_func", "exc"),
    [
        # no lower/upper specified
        (lambda x: x.clip(), ValueError),
        # out of range on quantile
        (lambda x: x.quantile(5.0), ValueError),
    ],
)
def test_arraylike_functions_transform_errors(t, ibis_func, exc):
    with pytest.raises(exc):
        ibis_func(t.float64_with_zeros).execute()


def test_quantile_multi_array_access(client, t, df):
    quantile = t.float64_with_zeros.quantile([0.25, 0.5])
    expr = quantile[0], quantile[1]
    result = tuple(map(client.execute, expr))
    expected = tuple(df.float64_with_zeros.quantile([0.25, 0.5]))
    assert result == expected


@pytest.mark.parametrize(
    (
        "left",
        "right",
        "expected_value",
        "expected_type",
        "left_dtype",
        "right_dtype",
    ),
    [
        (True, 1, True, bool, dt.boolean, dt.int64),
        (True, 1.0, True, bool, dt.boolean, dt.float64),
        (True, True, True, bool, dt.boolean, dt.boolean),
        (False, 0, False, bool, dt.boolean, dt.int64),
        (False, 0.0, False, bool, dt.boolean, dt.float64),
        (False, False, False, bool, dt.boolean, dt.boolean),
        (1, True, 1, int, dt.int64, dt.boolean),
        (1, 1.0, 1, int, dt.int64, dt.float64),
        (1, 1, 1, int, dt.int64, dt.int64),
        (0, False, 0, int, dt.int64, dt.boolean),
        (0, 0.0, 0, int, dt.int64, dt.float64),
        (0, 0, 0, int, dt.int64, dt.int64),
        (1.0, True, 1.0, float, dt.float64, dt.boolean),
        (1.0, 1, 1.0, float, dt.float64, dt.int64),
        (1.0, 1.0, 1.0, float, dt.float64, dt.float64),
        (0.0, False, 0.0, float, dt.float64, dt.boolean),
        (0.0, 0, 0.0, float, dt.float64, dt.int64),
        (0.0, 0.0, 0.0, float, dt.float64, dt.float64),
    ],
)
def test_execute_with_same_hash_value_in_scope(
    left, right, expected_value, expected_type, left_dtype, right_dtype
):
    @udf.elementwise([left_dtype, right_dtype], left_dtype)
    def my_func(x, _):
        return x

    df = pd.DataFrame({"left": [left], "right": [right]})
    table = ibis.pandas.connect().from_dataframe(df)

    expr = my_func(table.left, table.right)
    result = execute(expr.op())
    assert isinstance(result, pd.Series)

    result = result.tolist()
    assert result == [expected_value]
    assert type(result[0]) is expected_type


def test_ifelse_returning_bool():
    one = ibis.literal(1)
    two = ibis.literal(2)
    true = ibis.literal(True)
    false = ibis.literal(False)
    expr = ibis.ifelse(one + one == two, true, false)
    result = execute(expr.op())
    assert result is True


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        pytest.param(dt.float64, 1, id="float_int"),
        pytest.param(dt.float64, True, id="float_bool"),
        pytest.param(dt.int64, 1.0, id="int_float"),
        pytest.param(dt.int64, True, id="int_bool"),
        pytest.param(dt.boolean, 1.0, id="bool_float"),
        pytest.param(dt.boolean, 1, id="bool_int"),
    ],
)
def test_signature_does_not_match_input_type(dtype, value):
    @udf.elementwise([dtype], dtype)
    def func(x):
        return x

    df = pd.DataFrame({"col": [value]})
    table = ibis.pandas.connect().from_dataframe(df)

    result = execute(table.col.op())
    assert isinstance(result, pd.Series)

    result = result.tolist()
    assert result == [value]
    assert type(result[0]) is type(value)


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        (
            lambda x: x.approx_median(),
            lambda x: x.median(),
        )
    ],
)
@pytest.mark.parametrize("column", ["float64_with_zeros", "int64_with_zeros"])
def test_approx_median(t, df, ibis_func, pandas_func, column):
    expr = ibis_func(t[column])
    result = expr.execute()
    expected = pandas_func(df[column])
    assert expected == result
