from __future__ import annotations

import decimal
import operator
from functools import reduce, singledispatch

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newutils import asframe, asseries, columnwise
from ibis.backends.pandas.rewrites import (
    PandasAggregate,
    PandasJoin,
    PandasReduce,
    PandasRename,
)
from ibis.common.exceptions import OperationNotDefinedError
from ibis.formats.pandas import PandasData
from ibis.util import gen_name

################## PANDAS SPECIFIC NODES ######################


@singledispatch
def execute(node, **kwargs):
    raise OperationNotDefinedError(f"no rule for {type(node)}")


@execute.register(ops.Literal)
def execute_literal(op, value, dtype):
    if dtype.is_interval():
        value = pd.Timedelta(value, dtype.unit.short)
    elif dtype.is_array():
        value = np.array(value)
    elif dtype.is_date():
        value = pd.Timestamp(value, tz="UTC").tz_localize(None)

    return value


@execute.register(ops.DatabaseTable)
def execute_database_table(op, name, schema, source, namespace):
    return source.dictionary[name]


@execute.register(ops.InMemoryTable)
def execute_in_memory_table(op, name, schema, data):
    return data.to_frame()


@execute.register(ops.DummyTable)
def execute_dummy_table(op, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Limit)
def execute_limit(op, parent, n, offset):
    if n is None:
        return parent.iloc[offset:]
    else:
        return parent.iloc[offset : offset + n]


@execute.register(ops.Sample)
def execute_sample(op, parent, fraction, method, seed):
    return parent.sample(frac=fraction, random_state=seed)


@execute.register(ops.Filter)
def execute_filter(op, parent, predicates):
    if predicates:
        pred = reduce(operator.and_, predicates)
        if len(pred) != len(parent):
            raise RuntimeError(
                "Selection predicate length does not match underlying table"
            )
        parent = parent.loc[pred].reset_index(drop=True)
    return parent


@execute.register(PandasRename)
def execute_rename(op, parent, mapping):
    return parent.rename(columns=mapping)


@execute.register(PandasJoin)
def execute_join(op, left, right, left_on, right_on, how):
    # broadcast predicates if they are scalar values
    left_size = len(left)
    left_on = [asseries(v, left_size) for v in left_on]
    right_size = len(right)
    right_on = [asseries(v, right_size) for v in right_on]

    if how == "cross":
        assert not left_on and not right_on
        return pd.merge(left, right, how="cross")
    elif how == "anti":
        df = pd.merge(
            left, right, how="outer", left_on=left_on, right_on=right_on, indicator=True
        )
        df = df[df["_merge"] == "left_only"]
        return df.drop(columns=["_merge"])
    elif how == "semi":
        mask = asseries(True, left_size)
        for left_pred, right_pred in zip(left_on, right_on):
            mask = mask & left_pred.isin(right_pred)
        return left[mask]
    elif how == "asof":
        df = pd.merge_asof(left, right, left_on=left_on, right_on=right_on)
        return df
    else:
        df = left.merge(right, how=how, left_on=left_on, right_on=right_on)
        return df.drop(columns=[f"key_{i}" for i in range(len(left_on))])


@execute.register(ops.Union)
def execute_union(op, left, right, distinct):
    result = pd.concat([left, right], axis=0)
    return result.drop_duplicates() if distinct else result


@execute.register(ops.Intersection)
def execute_intersection_dataframe_dataframe(op, left, right, distinct):
    if not distinct:
        raise NotImplementedError(
            "`distinct=False` is not supported by the pandas backend"
        )
    return left.merge(right, on=list(left.columns), how="inner")


@execute.register(ops.Distinct)
def execute_distinct(op, parent):
    return parent.drop_duplicates()


@execute.register(ops.Project)
def execute_project(op, parent, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Sort)
def execute_sort(op, parent, keys):
    # 1. add sort key columns to the dataframe if they are not already present
    # 2. sort the dataframe using those columns
    # 3. drop the sort key columns

    names = []
    newcols = {}
    for key, keycol in zip(op.keys, keys):
        if not isinstance(key, ops.Field):
            name = gen_name("sort_key")
            newcols[name] = keycol
        names.append(name)

    result = parent.assign(**newcols)
    ascending = [key.ascending for key in op.keys]
    result = result.sort_values(by=names, ascending=ascending, ignore_index=True)

    return result.drop(newcols.keys(), axis=1)


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
    if isinstance(default, pd.Series):
        raise NotImplementedError(
            "SearchedCase with a columnar shaped default value is not implemented"
        )

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


@execute.register(ops.DropNa)
def execute_dropna(op, parent, how, subset):
    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None
    return parent.dropna(how=how, subset=subset)


@execute.register(PandasReduce)
def execute_pandas_reduce(op, parent, metrics):
    results = {k: v(parent) for k, v in metrics.items()}
    combined, _ = asframe(results)
    return combined


@execute.register(PandasAggregate)
def execute_pandas_aggregate(op, parent, groups, metrics):
    parent = parent.groupby([name for name, col in groups.items()])
    metrics = {k: parent.apply(v) for k, v in metrics.items()}

    result = pd.concat(metrics, axis=1).reset_index()
    return result


@execute.register(ops.InValues)
def execute_in_values(op, value, options):
    if isinstance(value, pd.Series):
        return value.isin(options)
    else:
        return value in options


@execute.register(ops.InSubquery)
def execute_in_subquery(op, rel, needle):
    first_column = rel.iloc[:, 0]
    if isinstance(needle, pd.Series):
        return needle.isin(first_column)
    else:
        return needle in first_column


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


def zuper(node, params):
    from ibis.backends.pandas.rewrites import (
        rewrite_aggregate,
        rewrite_join,
        rewrite_project,
    )
    from ibis.expr.rewrites import p

    replace_literals = p.ScalarParameter >> (
        lambda _: ops.Literal(value=params[_], dtype=_.dtype)
    )

    def fn(node, _, **kwargs):
        # TODO(kszucs): need to clean up the resultset as soon as an intermediate
        # result is not needed anymore
        result = execute(node, **kwargs)
        return result

    original = node

    node = node.to_expr().as_table().op()
    node = node.replace(
        rewrite_project | rewrite_aggregate | rewrite_join | replace_literals
    )
    # print(node.to_expr())
    df = node.map(fn)[node]

    # TODO(kszucs): add a flag to disable this conversion because it can be
    # expensive for columns with object dtype
    df = PandasData.convert_table(df, node.schema)
    if isinstance(original, ops.Value):
        if original.shape.is_scalar():
            return df.iloc[0, 0]
        elif original.shape.is_columnar():
            return df.iloc[:, 0]
        else:
            raise TypeError(f"Unexpected shape: {original.shape}")
    else:
        return df
