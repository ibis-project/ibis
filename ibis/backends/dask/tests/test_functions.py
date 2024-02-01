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
from ibis.backends.dask.tests.conftest import TestConf as tm


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
    expected = op(df.plain_float64, df.plain_int64).compute()
    tm.assert_series_equal(
        result.reset_index(drop=True).rename("tmp"),
        expected.reset_index(drop=True).rename("tmp"),
    )


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, pandas_df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.execute()
    expected = op(pandas_df.plain_int64 == 1, pandas_df.plain_int64 == 2)
    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


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
        param(
            methodcaller("exp"),
            methodcaller("exp"),
            id="exp",
            marks=pytest.mark.xfail(
                reason="Unable to normalize Decimal('2.71513316E+43') as decimal with precision 12 and scale 3",
                raises=TypeError,
            ),
        ),
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
def test_math_functions_decimal(t, pandas_df, ibis_func, pandas_func):
    dtype = dt.Decimal(12, 3)
    context = decimal.Context(prec=dtype.precision)
    p = decimal.Decimal(f"{'0' * (dtype.precision - dtype.scale)}.{'0' * dtype.scale}")

    def func(x):
        x = context.create_decimal(x)
        x = pandas_func(x)
        if math.isnan(x):
            return float("nan")
        return x.quantize(p)

    expr = ibis_func(t.float64_as_strings.cast(dtype))
    result = expr.execute()
    expected = pandas_df.float64_as_strings.map(func, na_action="ignore")
    tm.assert_series_equal(result, expected, check_names=False)


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
    ("ibis_func", "dask_func"),
    [
        (
            lambda x: x.quantile([0.25, 0.75]),
            lambda x: list(x.quantile([0.25, 0.75])),
        )
    ],
)
@pytest.mark.parametrize("column", ["float64_with_zeros", "int64_with_zeros"])
def test_quantile_list(t, pandas_df, ibis_func, dask_func, column):
    expr = ibis_func(t[column])
    result = expr.execute()
    expected = dask_func(pandas_df[column])
    assert result == expected


@pytest.mark.parametrize(
    ("ibis_func", "dask_func"),
    [
        (lambda x: x.quantile(0), lambda x: x.quantile(0)),
        (lambda x: x.quantile(1), lambda x: x.quantile(1)),
        (
            lambda x: x.quantile(0.5),
            lambda x: x.quantile(0.5),
        ),
    ],
)
def test_quantile_scalar(t, pandas_df, ibis_func, dask_func):
    result = ibis_func(t.float64_with_zeros).execute()
    expected = dask_func(pandas_df.float64_with_zeros)
    assert result == expected

    result = ibis_func(t.int64_with_zeros).execute()
    expected = dask_func(pandas_df.int64_with_zeros)
    assert result == expected


@pytest.mark.parametrize(
    ("ibis_func", "exc"),
    [
        # no lower/upper specified
        (lambda x: x.clip(), ValueError),
        # out of range on quantile
        (lambda x: x.quantile(5.0), ValueError),
    ],
)
def test_arraylike_functions_transform_errors(t, df, ibis_func, exc):
    with pytest.raises(exc):
        ibis_func(t.float64_with_zeros).execute()


def test_ifelse_returning_bool(con):
    one = ibis.literal(1)
    two = ibis.literal(2)
    true = ibis.literal(True)
    false = ibis.literal(False)
    expr = ibis.ifelse(one + one == two, true, false)
    result = con.execute(expr)
    assert result is np.bool_(True)
