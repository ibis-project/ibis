from __future__ import annotations

import operator

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.executor.core import execute
from ibis.common.exceptions import OperationNotDefinedError

# these are serieswise functions
_reduction_functions = {
    ops.Min: lambda x: x.min(),
    ops.Max: lambda x: x.max(),
    ops.Sum: lambda x: x.sum(),
    ops.Mean: lambda x: x.mean(),
    ops.Count: lambda x: x.count(),
    ops.Mode: lambda x: x.mode().iloc[0],
    ops.Any: lambda x: x.any(),
    ops.All: lambda x: x.all(),
    ops.Median: lambda x: x.median(),
    ops.ApproxMedian: lambda x: x.median(),
    ops.BitAnd: lambda x: np.bitwise_and.reduce(x.values),
    ops.BitOr: lambda x: np.bitwise_or.reduce(x.values),
    ops.BitXor: lambda x: np.bitwise_xor.reduce(x.values),
    ops.Last: lambda x: x.iloc[-1],
    ops.First: lambda x: x.iloc[0],
    ops.CountDistinct: lambda x: x.nunique(),
    ops.ApproxCountDistinct: lambda x: x.nunique(),
}


# could columnwise be used here?
def agg(func, arg_column, where_column):
    if where_column is None:

        def applier(df):
            return func(df[arg_column.name])
    else:

        def applier(df):
            mask = df[where_column.name]
            col = df[arg_column.name][mask]
            return func(col)

    return applier


@execute.register(ops.Reduction)
def execute_reduction(op, arg, where):
    func = _reduction_functions[type(op)]
    return agg(func, arg, where)


variance_ddof = {"pop": 0, "sample": 1}


@execute.register(ops.Variance)
def execute_variance(op, arg, where, how):
    ddof = variance_ddof[how]
    return agg(lambda x: x.var(ddof=ddof), arg, where)


@execute.register(ops.StandardDev)
def execute_standard_dev(op, arg, where, how):
    ddof = variance_ddof[how]
    return agg(lambda x: x.std(ddof=ddof), arg, where)


@execute.register(ops.Correlation)
def execute_correlation(op, left, right, where, how):
    if where is None:

        def agg(df):
            return df[left.name].corr(df[right.name])
    else:

        def agg(df):
            mask = df[where.name]
            lhs = df[left.name][mask]
            rhs = df[right.name][mask]
            return lhs.corr(rhs)

    return agg


@execute.register(ops.Covariance)
def execute_covariance(op, left, right, where, how):
    ddof = variance_ddof[how]
    if where is None:

        def agg(df):
            return df[left.name].cov(df[right.name], ddof=ddof)
    else:

        def agg(df):
            mask = df[where.name]
            lhs = df[left.name][mask]
            rhs = df[right.name][mask]
            return lhs.cov(rhs, ddof=ddof)

    return agg


@execute.register(ops.ArgMin)
@execute.register(ops.ArgMax)
def execute_argminmax(op, arg, key, where):
    func = operator.methodcaller(op.__class__.__name__.lower())

    if where is None:

        def agg(df):
            indices = func(df[key.name])
            return df[arg.name].iloc[indices]
    else:

        def agg(df):
            mask = df[where.name]
            filtered = df[mask]
            indices = func(filtered[key.name])
            return filtered[arg.name].iloc[indices]

    return agg


@execute.register(ops.ArrayCollect)
def execute_array_collect(op, arg, where):
    if where is None:

        def agg(df):
            return df[arg.name].tolist()
    else:

        def agg(df):
            mask = df[where.name]
            return df[arg.name][mask].tolist()

    return agg


@execute.register(ops.GroupConcat)
def execute_group_concat(op, arg, sep, where):
    if where is None:

        def agg(df):
            return sep.join(df[arg.name].astype(str))
    else:

        def agg(df):
            mask = df[where.name]
            group = df[arg.name][mask]
            if group.empty:
                return pd.NA
            return sep.join(group)

    return agg


@execute.register(ops.Quantile)
def execute_quantile(op, arg, quantile, where):
    return agg(lambda x: x.quantile(quantile), arg, where)


@execute.register(ops.MultiQuantile)
def execute_multi_quantile(op, arg, quantile, where):
    return agg(lambda x: list(x.quantile(quantile)), arg, where)


@execute.register(ops.Arbitrary)
def execute_arbitrary(op, arg, where, how):
    # TODO(kszucs): could be rewritten to ops.Last and ops.First prior to execution
    if how == "first":
        return agg(lambda x: x.iloc[0], arg, where)
    elif how == "last":
        return agg(lambda x: x.iloc[-1], arg, where)
    else:
        raise OperationNotDefinedError(f"Arbitrary {how!r} is not supported")


@execute.register(ops.CountStar)
def execute_count_star(op, arg, where):
    # TODO(kszucs): revisit the arg handling here
    def agg(df):
        if where is None:
            return len(df)
        else:
            return df[where.name].sum()

    return agg


@execute.register(ops.CountDistinctStar)
def execute_count_distinct_star(op, arg, where):
    def agg(df):
        if where is None:
            return df.nunique()
        else:
            return df[where.name].nunique()

    return agg


@execute.register(ops.ReductionVectorizedUDF)
def execute_reduction_udf(op, func, func_args, input_type, return_type):
    """Execute a reduction UDF."""

    def agg(df):
        args = [df[col.name] for col in func_args]
        return func(*args)

    return agg
