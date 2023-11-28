from __future__ import annotations

import operator
from functools import reduce, singledispatch

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis import util
from ibis.formats.pandas import PandasData, PandasSchema, PandasType

################## PANDAS SPECIFIC NODES ######################

# class PandasRelation(ops.Relation):
#     pass


# class SelectColumns(PandasRelation):
#     fields


################## REWRITE RULES ##############################

# @replace(ops.Project)
# def rewrite_project(expr):
#     # 1. gather plain column references and construct an operation picking those
#     #    columns from the table
#     # 2. collect the expressions which need to be computed
#     # 3. merge the dataframe from 1. with the expressions from 2.
#     pass

# @replace(ops.JoinChain)
# def rewrite_join_chain(expr):
#     # explode the join chain into a nested join tree
#     pass

###############################################################


@singledispatch
def execute(node, **kwargs):
    raise NotImplementedError(f"no rule for {type(node)}")


@execute.register(ops.Literal)
def execute_literal(op, value, dtype):
    if dtype.is_interval():
        return pd.Timedelta(value, dtype.unit.short)
    elif value is None:
        return value
    elif dtype.is_array():
        return np.array(value)
    else:
        return value


@execute.register(ops.DatabaseTable)
def execute_database_table(op, name, schema, source, namespace):
    return source.dictionary[name]
    # if timecontext:
    #     begin, end = timecontext
    #     time_col = get_time_col()
    #     if time_col not in df:
    #         raise com.IbisError(
    #             f"Table {op.name} must have a time column named {time_col}"
    #             " to execute with time context."
    #         )
    #     # filter with time context
    #     mask = df[time_col].between(begin, end)
    #     return df.loc[mask].reset_index(drop=True)


@execute.register(ops.InMemoryTable)
def execute_in_memory_table(op, name, schema, data):
    return data.to_frame()


@execute.register(ops.Limit)
def execute_limit(op, parent, n, offset):
    if n is None:
        return parent.iloc[offset:]
    else:
        return parent.iloc[offset : offset + n]


@execute.register(ops.Filter)
def execute_filter(op, parent, predicates):
    if predicates:
        predicate = reduce(operator.and_, predicates)
        assert len(predicate) == len(
            parent
        ), "Selection predicate length does not match underlying table"
        parent = parent.loc[predicate]
    return parent


@execute.register(ops.Project)
def execute_project(op, parent, values):
    return pd.DataFrame(values)


@execute.register(ops.Sort)
def execute_sort(op, parent, keys):
    # 1. add sort key columns to the dataframe
    # 2. sort the dataframe using those columns
    # 3. drop the sort key columns

    columns = {util.gen_name("sort_key"): key for key in keys}
    result = parent.assign(**columns)

    by = list(columns.keys())
    ascending = [key.ascending for key in op.keys]
    result = result.sort_values(by=by, ascending=ascending)

    return result.drop(by, axis=1)


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
        raise NotImplementedError(f"no rule for {type(arg)}")


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
    ops.BitwiseLeftShift: lambda x, y: np.left_shift(x, y),
    ops.BitwiseRightShift: lambda x, y: np.right_shift(x, y),
}


@execute.register(ops.Binary)
def execute_equals(op, left, right):
    return _binary_operations[type(op)](left, right)


@execute.register(ops.IfElse)
def execute_if_else(op, bool_expr, true_expr, false_null_expr):
    """Execute `where` following ibis's intended semantics."""
    cond = bool_expr
    true = true_expr
    false = false_null_expr
    if isinstance(cond, pd.Series):
        if not isinstance(true, pd.Series):
            true = pd.Series(
                np.repeat(true, len(cond)), name=cond.name, index=cond.index
            )
        return true.where(cond, other=false)
    if cond:
        if isinstance(false, pd.Series) and not isinstance(true, pd.Series):
            return pd.Series(np.repeat(true, len(false)))
        return true
    else:
        if isinstance(true, pd.Series) and not isinstance(false, pd.Series):
            return pd.Series(np.repeat(false, len(true)), index=true.index)
        return false


@execute.register(ops.NullIf)
def execute_null_if(op, arg, null_if_expr):
    if isinstance(arg, pd.Series):
        return arg.where(arg != null_if_expr)
    elif isinstance(null_if_expr, pd.Series):
        return null_if_expr.where(arg != null_if_expr)
    else:
        return np.nan if arg == null_if_expr else arg


@execute.register(ops.FillNa)
def execute_fillna(op, parent, replacements):
    return parent.fillna(replacements)


@execute.register(ops.DropNa)
def execute_dropna(op, parent, how, subset):
    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None
    return parent.dropna(how=how, subset=subset)


@execute.register(ops.Reduction)
def execute_reduction(op, arg, where):
    if where is not None:
        arg = arg[where[arg.index]]

    # op_type = type(op)
    # if op_type == ops.BitwiseNot:
    #     function = np.bitwise_not
    # else:
    #     function = getattr(np, op_type.__name__.lower())
    # return call_numpy_ufunc(function, op, data, **kwargs)
    name = op.__class__.__name__.lower()
    method = getattr(arg, name)
    return method()


@execute.register(ops.CountStar)
def execute_count_star(op, arg, where):
    if where is None:
        return len(arg)  # arg.size() if arg DataFrameGroupBy
    else:
        return len(arg) - len(where) + where.sum()


@execute.register(ops.InValues)
def execute_in_values(op, value, options):
    if isinstance(value, pd.Series):
        return value.isin(options)
    # elif isinstance(value, SeriesGroupBy):
    #     return data.obj.isin(elements).groupby(
    #         get_grouping(data.grouper.groupings), group_keys=False
    #     )
    else:
        return value in options


@execute.register(ops.StringLength)
def execute_string_length(op, arg):
    return arg.str.len().astype("int32")


@execute.register(ops.StringReplace)
def execute_string_replace(op, arg, pattern, replacement):
    return arg.str.replace(pattern, replacement)


@execute.register(ops.Date)
def execute_date(op, arg):
    return arg.dt.floor("d")


@execute.register(ops.TimestampNow)
def execute_timestamp_now(op, *args, **kwargs):
    # timecontext = kwargs.get("timecontext", None)
    return pd.Timestamp("now", tz="UTC").tz_localize(None)


@execute.register(ops.TimestampDiff)
def execute_timestamp_diff(op, left, right):
    return left - right


def zuper(node):
    def fn(node, _, **kwargs):
        result = execute(node, **kwargs)
        return result

    result = node.map(fn)[node]
    return result
