"""Execution rules for generic ibis operations."""

from __future__ import annotations

import contextlib
import datetime
import decimal
import functools
import numbers
from operator import methodcaller

import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import pandas as pd
from pandas import isnull, to_datetime
from pandas.api.types import DatetimeTZDtype

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util
from ibis.backends.dask import Backend as DaskBackend
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    TypeRegistrationDict,
    add_globally_consecutive_column,
    make_selected_obj,
    register_types_to_dispatcher,
    rename_index,
)
from ibis.backends.pandas.core import (
    date_types,
    integer_types,
    numeric_types,
    scalar_types,
    simple_types,
    timestamp_types,
)
from ibis.backends.pandas.execution import constants
from ibis.backends.pandas.execution.generic import (
    _execute_binary_op_impl,
    compute_row_reduction,
    execute_between,
    execute_cast_series_array,
    execute_cast_series_generic,
    execute_count_distinct_star_frame,
    execute_count_distinct_star_frame_filter,
    execute_count_star_frame,
    execute_count_star_frame_filter,
    execute_count_star_frame_groupby,
    execute_database_table_client,
    execute_difference_dataframe_dataframe,
    execute_distinct_dataframe,
    execute_intersection_dataframe_dataframe,
    execute_isinf,
    execute_isnan,
    execute_node_column_in_column,
    execute_node_column_in_values,
    execute_node_dropna_dataframe,
    execute_node_fillna_dataframe_dict,
    execute_node_fillna_dataframe_scalar,
    execute_node_nullif_scalar_series,
    execute_node_nullif_series,
    execute_node_self_reference_dataframe,
    execute_searched_case,
    execute_series_clip,
    execute_series_isnull,
    execute_series_notnnull,
    execute_sort_key_series,
    execute_table_column_df_or_df_groupby,
)

# Many dask and pandas functions are functionally equivalent, so we just add
# on registrations for dask types
DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.Cast: [
        ((dd.Series, dt.DataType), execute_cast_series_generic),
        ((dd.Series, dt.Array), execute_cast_series_array),
    ],
    ops.SortKey: [((dd.Series, bool), execute_sort_key_series)],
    ops.Clip: [
        (
            (
                dd.Series,
                (dd.Series, type(None)) + numeric_types,
                (dd.Series, type(None)) + numeric_types,
            ),
            execute_series_clip,
        ),
    ],
    ops.TableColumn: [
        (
            ((dd.DataFrame, ddgb.DataFrameGroupBy),),
            execute_table_column_df_or_df_groupby,
        ),
    ],
    ops.CountStar: [
        (
            (ddgb.DataFrameGroupBy, type(None)),
            execute_count_star_frame_groupby,
        ),
        ((dd.DataFrame, type(None)), execute_count_star_frame),
        ((dd.DataFrame, dd.Series), execute_count_star_frame_filter),
    ],
    ops.CountDistinctStar: [
        (
            (ddgb.DataFrameGroupBy, type(None)),
            execute_count_star_frame_groupby,
        ),
        ((dd.DataFrame, type(None)), execute_count_distinct_star_frame),
        ((dd.DataFrame, dd.Series), execute_count_distinct_star_frame_filter),
    ],
    ops.Between: [
        (
            (
                dd.Series,
                (dd.Series, numbers.Real, str, datetime.datetime),
                (dd.Series, numbers.Real, str, datetime.datetime),
            ),
            execute_between,
        ),
    ],
    ops.Intersection: [
        (
            (dd.DataFrame, dd.DataFrame, bool),
            execute_intersection_dataframe_dataframe,
        )
    ],
    ops.Difference: [
        (
            (dd.DataFrame, dd.DataFrame, bool),
            execute_difference_dataframe_dataframe,
        )
    ],
    ops.DropNa: [((dd.DataFrame,), execute_node_dropna_dataframe)],
    ops.FillNa: [
        ((dd.DataFrame, simple_types), execute_node_fillna_dataframe_scalar),
        ((dd.DataFrame,), execute_node_fillna_dataframe_dict),
    ],
    ops.IsNull: [((dd.Series,), execute_series_isnull)],
    ops.NotNull: [((dd.Series,), execute_series_notnnull)],
    ops.IsNan: [((dd.Series,), execute_isnan)],
    ops.IsInf: [((dd.Series,), execute_isinf)],
    ops.SelfReference: [((dd.DataFrame,), execute_node_self_reference_dataframe)],
    ops.InValues: [((dd.Series, tuple), execute_node_column_in_values)],
    ops.InColumn: [((dd.Series, dd.Series), execute_node_column_in_column)],
    ops.NullIf: [
        ((dd.Series, (dd.Series, *simple_types)), execute_node_nullif_series),
        ((simple_types, dd.Series), execute_node_nullif_scalar_series),
    ],
    ops.Distinct: [((dd.DataFrame,), execute_distinct_dataframe)],
}

register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)

execute_node.register(ops.DatabaseTable, DaskBackend)(execute_database_table_client)


@execute_node.register(ops.Alias, object)
def execute_alias_series(op, _, **kwargs):
    # just compile the underlying argument because the naming is handled
    # by the translator for the top level expression
    return execute(op.arg, **kwargs)


@execute_node.register(ops.Arbitrary, dd.Series, (dd.Series, type(None)))
def execute_arbitrary_series_mask(op, data, mask, aggcontext=None, **kwargs):
    """Execute a masked `ops.Arbitrary` operation.

    We cannot use the pandas version because
    [Dask does not support `.iloc`](https://docs.dask.org/en/latest/dataframe-indexing.html).
    `.loc` will only work if our index lines up with the label.
    """
    data = data[mask] if mask is not None else data
    if op.how == "first":
        index = 0
    elif op.how == "last":
        index = len(data) - 1  # TODO - computation
    else:
        raise com.OperationNotDefinedError(f"Arbitrary {op.how!r} is not supported")

    return data.loc[index]


@execute_node.register(ops.Arbitrary, ddgb.SeriesGroupBy, type(None))
def execute_arbitrary_series_groupby(op, data, _, aggcontext=None, **kwargs):
    how = op.how
    if how is None:
        how = "first"

    if how not in {"first", "last"}:
        raise com.OperationNotDefinedError(f"Arbitrary {how!r} is not supported")
    return aggcontext.agg(data, how)


def _mode_agg(df):
    return df.sum().sort_values(ascending=False).index[0]


@execute_node.register(ops.Mode, dd.Series, (dd.Series, type(None)))
def execute_mode_series(_, data, mask, **kwargs):
    if mask is not None:
        data = data[mask]
    return data.reduction(
        chunk=methodcaller("value_counts"),
        combine=methodcaller("sum"),
        aggregate=_mode_agg,
        meta=data.dtype,
    )


def _grouped_mode_agg(gb):
    return gb.obj.groupby(gb.obj.index.names).sum()


def _grouped_mode_finalize(series):
    counts = "__counts__"
    values = series.index.names[-1]
    df = series.reset_index(-1, name=counts)
    out = df.groupby(df.index.names).apply(
        lambda g: g.sort_values(counts, ascending=False).iloc[0]
    )
    return out[values]


@execute_node.register(ops.Mode, ddgb.SeriesGroupBy, (ddgb.SeriesGroupBy, type(None)))
def execute_mode_series_group_by(_, data, mask, **kwargs):
    if mask is not None:
        data = data[mask]
    return data.agg(
        dd.Aggregation(
            name="mode",
            chunk=methodcaller("value_counts"),
            agg=_grouped_mode_agg,
            finalize=_grouped_mode_finalize,
        )
    )


@execute_node.register(ops.Cast, ddgb.SeriesGroupBy, dt.DataType)
def execute_cast_series_group_by(op, data, type, **kwargs):
    result = execute_cast_series_generic(op, make_selected_obj(data), type, **kwargs)
    return result.groupby(data.index)


def cast_scalar_to_timestamp(data, tz):
    if isinstance(data, str):
        return pd.Timestamp(data, tz=tz)
    return pd.Timestamp(data, unit="s", tz=tz)


@execute_node.register(ops.Cast, dd.core.Scalar, dt.Timestamp)
def execute_cast_scalar_timestamp(op, data, type, **kwargs):
    return dd.map_partitions(
        cast_scalar_to_timestamp, data, tz=type.timezone, meta="datetime64[ns]"
    )


def cast_series_to_timestamp(data, tz):
    if pd.api.types.is_string_dtype(data):
        timestamps = to_datetime(data)
    else:
        timestamps = to_datetime(data, unit="s")
    if getattr(timestamps.dtype, "tz", None) is not None:
        return timestamps.dt.tz_convert(tz)
    return timestamps.dt.tz_localize(tz)


@execute_node.register(ops.Cast, dd.Series, dt.Timestamp)
def execute_cast_series_timestamp(op, data, type, **kwargs):
    arg = op.arg
    from_type = arg.dtype

    if from_type.equals(type):  # noop cast
        return data

    tz = type.timezone
    dtype = "M8[ns]" if tz is None else DatetimeTZDtype("ns", tz)

    if from_type.is_timestamp():
        from_tz = from_type.timezone
        if tz is None and from_tz is None:
            return data
        elif tz is None or from_tz is None:
            return data.dt.tz_localize(tz)
        elif tz is not None and from_tz is not None:
            return data.dt.tz_convert(tz)
    elif from_type.is_date():
        return data if tz is None else data.dt.tz_localize(tz)
    elif from_type.is_string() or from_type.is_integer():
        return data.map_partitions(
            cast_series_to_timestamp,
            tz,
            meta=(data.name, dtype),
        )

    raise TypeError(f"Don't know how to cast {from_type} to {type}")


@execute_node.register(ops.Cast, dd.Series, dt.Date)
def execute_cast_series_date(op, data, type, **kwargs):
    arg = op.args[0]
    from_type = arg.dtype

    if from_type.equals(type):
        return data

    # TODO - we return slightly different things depending on the branch
    # double check what the logic should be

    if from_type.is_timestamp():
        return data.dt.normalize()

    if from_type.equals(dt.string):
        # TODO - this is broken
        datetimes = data.map_partitions(to_datetime, meta=(data.name, "datetime64[ns]"))

        # TODO - we are getting rid of the index here
        return datetimes.dt.normalize()

    if from_type.is_integer():
        return data.map_partitions(
            to_datetime, unit="D", meta=(data.name, "datetime64[ns]")
        )

    raise TypeError(f"Don't know how to cast {from_type} to {type}")


@execute_node.register(ops.Limit, dd.DataFrame, integer_types, integer_types)
def execute_limit_frame(op, data, nrows, offset, **kwargs):
    # NOTE: Dask Dataframes do not support iloc row based indexing
    # Need to add a globally consecutive index in order to select nrows number of rows
    if nrows == 0:
        return dd.from_pandas(
            pd.DataFrame(columns=data.columns).astype(data.dtypes), npartitions=1
        )
    unique_col_name = ibis.util.guid()
    df = add_globally_consecutive_column(data, col_name=unique_col_name)
    ret = df.loc[offset : (offset + nrows) - 1]
    return rename_index(ret, None)


@execute_node.register(ops.Limit, dd.DataFrame, type(None), integer_types)
def execute_limit_frame_no_limit(op, data, nrows, offset, **kwargs):
    unique_col_name = ibis.util.guid()
    df = add_globally_consecutive_column(data, col_name=unique_col_name)
    ret = df.loc[offset : (offset + len(df)) - 1]
    return rename_index(ret, None)


@execute_node.register(ops.Not, (dd.core.Scalar, dd.Series))
def execute_not_scalar_or_series(op, data, **kwargs):
    return ~data


@execute_node.register(ops.Binary, dd.Series, dd.Series)
@execute_node.register(ops.Binary, dd.Series, dd.core.Scalar)
@execute_node.register(ops.Binary, dd.core.Scalar, dd.Series)
@execute_node.register(ops.Binary, dd.core.Scalar, scalar_types)
@execute_node.register(ops.Binary, scalar_types, dd.core.Scalar)
@execute_node.register(ops.Binary, dd.core.Scalar, dd.core.Scalar)
@execute_node.register(
    (ops.NumericBinary, ops.LogicalBinary, ops.Comparison),
    numeric_types,
    dd.Series,
)
@execute_node.register(
    (ops.NumericBinary, ops.LogicalBinary, ops.Comparison),
    dd.Series,
    numeric_types,
)
@execute_node.register((ops.Comparison, ops.Add, ops.Multiply), dd.Series, str)
@execute_node.register((ops.Comparison, ops.Add, ops.Multiply), str, dd.Series)
@execute_node.register(ops.Comparison, dd.Series, timestamp_types)
@execute_node.register(ops.Comparison, timestamp_types, dd.Series)
@execute_node.register(ops.BitwiseBinary, integer_types, integer_types)
@execute_node.register(ops.BitwiseBinary, dd.Series, integer_types)
@execute_node.register(ops.BitwiseBinary, integer_types, dd.Series)
def execute_binary_op(op, left, right, **kwargs):
    return _execute_binary_op_impl(op, left, right, **kwargs)


@execute_node.register(ops.Comparison, dd.Series, date_types)
def execute_binary_op_date_right(op, left, right, **kwargs):
    return _execute_binary_op_impl(
        op, dd.to_datetime(left), pd.to_datetime(right), **kwargs
    )


@execute_node.register(ops.Binary, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_binary_op_series_group_by(op, left, right, **kwargs):
    if left.index != right.index:
        raise ValueError(
            f"Cannot perform {type(op).__name__} operation on two series with "
            "different groupings"
        )
    result = execute_binary_op(
        op, make_selected_obj(left), make_selected_obj(right), **kwargs
    )
    return result.groupby(left.index)


@execute_node.register(ops.Binary, ddgb.SeriesGroupBy, simple_types)
def execute_binary_op_series_gb_simple(op, left, right, **kwargs):
    result = execute_binary_op(op, make_selected_obj(left), right, **kwargs)
    return result.groupby(left.index)


@execute_node.register(ops.Binary, simple_types, ddgb.SeriesGroupBy)
def execute_binary_op_simple_series_gb(op, left, right, **kwargs):
    result = execute_binary_op(op, left, make_selected_obj(right), **kwargs)
    return result.groupby(right.index)


@execute_node.register(ops.Unary, ddgb.SeriesGroupBy)
def execute_unary_op_series_gb(op, operand, **kwargs):
    result = execute_node(op, make_selected_obj(operand), **kwargs)
    return result.groupby(operand.index)


@execute_node.register(
    (ops.Log, ops.Round),
    ddgb.SeriesGroupBy,
    (numbers.Real, decimal.Decimal, type(None)),
)
def execute_log_series_gb_others(op, left, right, **kwargs):
    result = execute_node(op, make_selected_obj(left), right, **kwargs)
    return result.groupby(left.index)


@execute_node.register((ops.Log, ops.Round), ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_log_series_gb_series_gb(op, left, right, **kwargs):
    result = execute_node(
        op, make_selected_obj(left), make_selected_obj(right), **kwargs
    )
    return result.groupby(left.index)


@execute_node.register(ops.Union, dd.DataFrame, dd.DataFrame, bool)
def execute_union_dataframe_dataframe(
    op, left: dd.DataFrame, right: dd.DataFrame, distinct, **kwargs
):
    result = dd.concat([left, right], axis=0)
    return result.drop_duplicates() if distinct else result


@execute_node.register(ops.NullIf, simple_types, dd.Series)
def execute_node_nullif_scalar_series(op, value, series, **kwargs):
    return series.where(series != value)


def wrap_case_result(raw: np.ndarray, expr: ir.Value):
    """Wrap a CASE statement result in a Series and handle returning scalars.

    Parameters
    ----------
    raw : ndarray[T]
        The raw results of executing the ``CASE`` expression
    expr : Value
        The expression from the which `raw` was computed

    Returns
    -------
    Union[scalar, Series]
    """
    raw_1d = np.atleast_1d(raw)
    if np.any(isnull(raw_1d)):
        result = dd.from_array(raw_1d)
    else:
        result = dd.from_array(
            raw_1d.astype(constants.IBIS_TYPE_TO_PANDAS_TYPE[expr.type()])
        )
    # TODO - we force computation here
    if isinstance(expr, ir.Scalar) and result.size.compute() == 1:
        return result.head().item()
    return result


@execute_node.register(ops.SearchedCase, tuple, tuple, object)
def execute_searched_case_dask(op, when_nodes, then_nodes, otherwise, **kwargs):
    whens = [execute(arg, **kwargs) for arg in when_nodes]
    thens = [execute(arg, **kwargs) for arg in then_nodes]
    if not isinstance(whens[0], dd.Series):
        # if we are not dealing with dask specific objects, fallback to the
        # pandas logic. For example, in the case of ibis literals.
        # See `test_functions/test_ifelse_returning_bool` or
        # `test_operations/test_searched_case_scalar` for code that hits this.
        return execute_searched_case(op, when_nodes, then_nodes, otherwise, **kwargs)

    if otherwise is None:
        otherwise = np.nan
    idx = whens[0].index
    whens = [w.to_dask_array() for w in whens]
    if isinstance(thens[0], dd.Series):
        # some computed column
        thens = [t.to_dask_array() for t in thens]
    else:
        # scalar
        thens = [da.from_array(np.array([t])) for t in thens]
    raw = da.select(whens, thens, otherwise)
    out = dd.from_dask_array(
        raw,
        index=idx,
    )
    return out


@execute_node.register(ops.SimpleCase, dd.Series, tuple, tuple, object)
def execute_simple_case_series(op, value, whens, thens, otherwise, **kwargs):
    whens = [execute(arg, **kwargs) for arg in whens]
    thens = [execute(arg, **kwargs) for arg in thens]
    if otherwise is None:
        otherwise = np.nan
    raw = np.select([value == when for when in whens], thens, otherwise)
    return wrap_case_result(raw, op.to_expr())


@execute_node.register(ops.Greatest, tuple)
def execute_node_greatest_list(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return compute_row_reduction(np.maximum.reduce, values, axis=0)


@execute_node.register(ops.Least, tuple)
def execute_node_least_list(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return compute_row_reduction(np.minimum.reduce, values, axis=0)


def coalesce(values):
    def reducer(a1, a2):
        with contextlib.suppress(AttributeError):
            a1 = a1.compute()
        return np.where(pd.isnull(a1), a2, a1)

    return functools.reduce(reducer, values)


@execute_node.register(ops.Coalesce, tuple)
def execute_node_coalesce(op, values, **kwargs):
    # TODO: this is slow
    values = [execute(arg, **kwargs) for arg in values]
    return compute_row_reduction(coalesce, values)


@execute_node.register(ops.TableArrayView, dd.DataFrame)
def execute_table_array_view(op, _, **kwargs):
    # Need to compute dataframe in order to squeeze into a scalar
    ddf = execute(op.table)
    return ddf.compute().squeeze()


@execute_node.register(ops.Sample, dd.DataFrame, object, object)
def execute_sample(op, data, fraction, seed, **kwargs):
    return data.sample(frac=fraction, random_state=seed)
