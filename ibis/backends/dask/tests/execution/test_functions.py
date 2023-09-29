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
from ibis.common.exceptions import OperationNotDefinedError

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask.execution import execute  # noqa: E402


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
    result = expr.compile()
    expected = op(df.plain_float64, df.plain_int64)
    tm.assert_series_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.compile()
    expected = op(df.plain_int64 == 1, df.plain_int64 == 2)
    tm.assert_series_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
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
    ("ibis_func", "dask_func"),
    [
        param(methodcaller("round"), lambda x: np.int64(round(x)), id="round"),
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
def test_math_functions_decimal(t, df, ibis_func, dask_func):
    dtype = dt.Decimal(12, 3)
    result = ibis_func(t.float64_as_strings.cast(dtype)).compile()
    context = decimal.Context(prec=dtype.precision)
    expected = df.float64_as_strings.apply(
        lambda x: context.create_decimal(x).quantize(
            decimal.Decimal(
                f"{'0' * (dtype.precision - dtype.scale)}.{'0' * dtype.scale}"
            )
        ),
        meta=("float64_as_strings", "object"),
    ).apply(dask_func, meta=("float64_as_strings", "object"))
    # dask.dataframe.Series doesn't do direct item assignment
    # TODO - maybe use .where instead
    computed_result = result.compute().reset_index(drop=True)
    computed_result[computed_result.apply(math.isnan)] = -99999
    computed_expected = expected.compute().reset_index(drop=True)
    computed_expected[computed_expected.apply(math.isnan)] = -99999
    # result[result.apply(math.isnan)] = -99999
    # expected[expected.apply(math.isnan)] = -99999
    tm.assert_series_equal(computed_result, computed_expected)


def test_round_decimal_with_negative_places(t, df):
    type = dt.Decimal(12, 3)
    expr = t.float64_as_strings.cast(type).round(-1)
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series(
            list(map(decimal.Decimal, ["1.0E+2", "2.3E+2", "-1.00E+3"])),
            name="float64_as_strings",
        ),
        npartitions=1,
    )
    tm.assert_series_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.xfail(
    raises=OperationNotDefinedError,
    reason="TODO - arrays - #2553"
    # Need an ops.MultiQuantile execution func that dispatches on ndarrays
)
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
def test_quantile_list(t, df, ibis_func, dask_func, column):
    expr = ibis_func(t[column])
    result = expr.compile()
    expected = dask_func(df[column])
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
def test_quantile_scalar(t, df, ibis_func, dask_func):
    result = ibis_func(t.float64_with_zeros).compile()
    expected = dask_func(df.float64_with_zeros)
    assert result.compute() == expected.compute()

    result = ibis_func(t.int64_with_zeros).compile()
    expected = dask_func(df.int64_with_zeros)
    assert result.compute() == expected.compute()


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


@pytest.mark.xfail(
    raises=OperationNotDefinedError,
    reason="TODO - arrays - #2553"
    # Need an ops.MultiQuantile execution func that dispatches on ndarrays
)
def test_quantile_array_access(client, t, df):
    quantile = t.float64_with_zeros.quantile([0.25, 0.5])
    expr = quantile[0], quantile[1]
    result = tuple(map(client.execute, expr))
    expected = tuple(df.float64_with_zeros.quantile([0.25, 0.5]))
    assert result == expected


def test_ifelse_returning_bool():
    one = ibis.literal(1)
    two = ibis.literal(2)
    true = ibis.literal(True)
    false = ibis.literal(False)
    expr = ibis.ifelse(one + one == two, true, false)
    result = execute(expr.op())
    assert result is True
