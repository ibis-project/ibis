from __future__ import annotations

import decimal
import operator

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.executor.core import execute
from ibis.backends.pandas.executor.utils import asframe, columnwise
from ibis.common.exceptions import OperationNotDefinedError
from ibis.formats.pandas import PandasData


@execute.register(ops.Literal)
def execute_literal(op, value, dtype):
    if dtype.is_interval():
        value = pd.Timedelta(value, dtype.unit.short)
    elif dtype.is_array():
        value = np.array(value)
    elif dtype.is_date():
        value = pd.Timestamp(value, tz="UTC").tz_localize(None)

    return value


@execute.register(ops.Field)
def execute_field(op, rel, name):
    return rel[name]


@execute.register(ops.Alias)
def execute_alias(op, arg, name):
    try:
        return arg.rename(name)
    except AttributeError:
        return arg


@execute.register(ops.SortKey)
def execute_sort_key(op, expr, ascending):
    return expr


@execute.register(ops.Not)
def execute_not(op, arg):
    if isinstance(arg, (bool, np.bool_)):
        return not arg
    else:
        return ~arg


@execute.register(ops.Negate)
def execute_negate(op, arg):
    if isinstance(arg, (bool, np.bool_)):
        return not arg
    else:
        return -arg


@execute.register(ops.Cast)
def execute_cast(op, arg, to):
    if isinstance(arg, pd.Series):
        return PandasData.convert_column(arg, to)
    else:
        return PandasData.convert_scalar(arg, to)


@execute.register(ops.E)
def execute_e(op):
    return np.e


@execute.register(ops.Pi)
def execute_pi(op):
    return np.pi


_unary_operations = {
    ops.Abs: abs,
    ops.Ceil: lambda x: np.ceil(x).astype("int64"),
    ops.Floor: lambda x: np.floor(x).astype("int64"),
    ops.Sign: np.sign,
    ops.Exp: np.exp,
    ops.Tan: np.tan,
    ops.Cos: np.cos,
    ops.Cot: lambda x: 1 / np.tan(x),
    ops.Sin: np.sin,
    ops.Atan: np.arctan,
    ops.Acos: np.arccos,
    ops.Asin: np.arcsin,
    ops.BitwiseNot: np.invert,
    ops.Radians: np.radians,
    ops.Degrees: np.degrees,
}

_binary_operations = {
    ops.Greater: operator.gt,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.GreaterEqual: operator.ge,
    ops.Equals: operator.eq,
    ops.NotEquals: operator.ne,
    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Modulus: operator.mod,
    ops.Power: operator.pow,
    ops.IdenticalTo: lambda x, y: (x == y) | (pd.isnull(x) & pd.isnull(y)),
    ops.BitwiseXor: lambda x, y: np.bitwise_xor(x, y),
    ops.BitwiseOr: lambda x, y: np.bitwise_or(x, y),
    ops.BitwiseAnd: lambda x, y: np.bitwise_and(x, y),
    ops.BitwiseLeftShift: lambda x, y: np.left_shift(x, y).astype("int64"),
    ops.BitwiseRightShift: lambda x, y: np.right_shift(x, y).astype("int64"),
    ops.Atan2: np.arctan2,
}


@execute.register(ops.Unary)
def execute_unary(op, arg):
    return _unary_operations[type(op)](arg)


@execute.register(ops.Binary)
def execute_equals(op, left, right):
    return _binary_operations[type(op)](left, right)


def mapdecimal(func, s):
    def wrapper(x):
        try:
            return func(x)
        except decimal.InvalidOperation:
            return decimal.Decimal("NaN")

    return s.map(wrapper)


@execute.register(ops.Sqrt)
def execute_sqrt(op, arg):
    if op.arg.dtype.is_decimal():
        return mapdecimal(lambda x: x.sqrt(), arg)
    else:
        return np.sqrt(arg)


@execute.register(ops.Ln)
def execute_ln(op, arg):
    if op.arg.dtype.is_decimal():
        return mapdecimal(lambda x: x.ln(), arg)
    else:
        return np.log(arg)


@execute.register(ops.Log2)
def execute_log2(op, arg):
    if op.arg.dtype.is_decimal():
        # TODO(kszucs): this doesn't support columnar shaped base
        baseln = decimal.Decimal(2).ln()
        return mapdecimal(lambda x: x.ln() / baseln, arg)
    else:
        return np.log2(arg)


@execute.register(ops.Log10)
def execute_log10(op, arg):
    if op.arg.dtype.is_decimal():
        return mapdecimal(lambda x: x.log10(), arg)
    else:
        return np.log10(arg)


@execute.register(ops.Log)
def execute_log(op, arg, base):
    if op.arg.dtype.is_decimal():
        # TODO(kszucs): this doesn't support columnar shaped base
        baseln = decimal.Decimal(base).ln()
        return mapdecimal(lambda x: x.ln() / baseln, arg)
    elif base is None:
        return np.log(arg)
    else:
        return np.log(arg) / np.log(base)


@execute.register(ops.Round)
def execute_round(op, arg, digits):
    if op.arg.dtype.is_decimal():
        if digits is None:
            return arg.map(round)
        else:
            return arg.map(lambda x: round(x, digits))

    elif digits is None:
        return np.round(arg).astype("int64")
    else:
        return np.round(arg, digits).astype("float64")


@execute.register(ops.Clip)
def execute_clip(op, **kwargs):
    return columnwise(
        lambda df: df["arg"].clip(lower=df["lower"], upper=df["upper"]), kwargs
    )


@execute.register(ops.IfElse)
def execute_if_else(op, **kwargs):
    return columnwise(
        lambda df: df["true_expr"].where(df["bool_expr"], other=df["false_null_expr"]),
        kwargs,
    )


@execute.register(ops.SearchedCase)
def execute_searched_case(op, cases, results, default):
    cases, _ = asframe(cases, concat=False)
    results, _ = asframe(results, concat=False)

    out = np.select(cases, results, default)
    return pd.Series(out)


@execute.register(ops.SimpleCase)
def execute_simple_case(op, base, cases, results, default):
    if isinstance(default, pd.Series):
        raise NotImplementedError(
            "SimpleCase with a columnar shaped default value is not implemented"
        )

    cases = tuple(base == case for case in cases)
    return execute_searched_case(op, cases, results, default)


@execute.register(ops.TypeOf)
def execute_typeof(op, arg):
    raise OperationNotDefinedError("TypeOf is not implemented")


@execute.register(ops.NullIf)
def execute_null_if(op, **kwargs):
    return columnwise(
        lambda df: df["arg"].where(df["arg"] != df["null_if_expr"]), kwargs
    )


@execute.register(ops.IsNull)
def execute_series_isnull(op, arg):
    return arg.isnull()


@execute.register(ops.NotNull)
def execute_series_notnnull(op, arg):
    return arg.notnull()


@execute.register(ops.FillNa)
def execute_fillna(op, parent, replacements):
    return parent.fillna(replacements)


@execute.register(ops.IsNan)
def execute_isnan(op, arg):
    try:
        return np.isnan(arg)
    except (TypeError, ValueError):
        # if `arg` contains `None` np.isnan will complain
        # so we take advantage of NaN not equaling itself
        # to do the correct thing
        return arg != arg


@execute.register(ops.IsInf)
def execute_isinf(op, arg):
    return np.isinf(arg)


@execute.register(ops.Greatest)
def execute_greatest(op, arg):
    return columnwise(lambda df: df.max(axis=1), arg)


@execute.register(ops.Least)
def execute_least(op, arg):
    return columnwise(lambda df: df.min(axis=1), arg)


@execute.register(ops.Coalesce)
def execute_coalesce(op, arg):
    return columnwise(lambda df: df.bfill(axis=1).iloc[:, 0], arg)


@execute.register(ops.Between)
def execute_between(op, arg, lower_bound, upper_bound):
    return arg.between(lower_bound, upper_bound)


@execute.register(ops.ElementWiseVectorizedUDF)
def execute_elementwise_udf(op, func, func_args, input_type, return_type):
    """Execute an elementwise UDF."""

    res = func(*func_args)
    if isinstance(res, pd.DataFrame):
        # it is important otherwise it is going to fill up the memory
        res = res.apply(lambda row: row.to_dict(), axis=1)

    return res
