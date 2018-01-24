from __future__ import absolute_import

import datetime
import decimal
import functools
import numbers
import operator

import six

import numpy as np
import pandas as pd

from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy

import toolz

from ibis import compat

import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.pandas.core import (
    integer_types, simple_types, numeric_types, fixed_width_types, scalar_types
)
from ibis.pandas.dispatch import execute, execute_node
from ibis.pandas.execution import constants


@execute_node.register(ir.Literal)
@execute_node.register(ir.Literal, object)
@execute_node.register(ir.Literal, object, dt.DataType)
def execute_node_literal(op, *args, **kwargs):
    return op.value


@execute_node.register(ops.Limit, pd.DataFrame, integer_types, integer_types)
def execute_limit_frame(op, data, limit, offset, **kwargs):
    return data.iloc[offset:offset + limit]


@execute_node.register(ops.Cast, SeriesGroupBy, dt.DataType)
def execute_cast_series_group_by(op, data, type, **kwargs):
    result = execute_node(op, data.obj, type, **kwargs)
    return result.groupby(data.grouper.groupings)


@execute_node.register(ops.Cast, pd.Series, dt.DataType)
def execute_cast_series_generic(op, data, type, **kwargs):
    return data.astype(constants.IBIS_TYPE_TO_PANDAS_TYPE[type])


@execute_node.register(ops.Cast, pd.Series, dt.Array)
def execute_cast_series_array(op, data, type, **kwargs):
    value_type = type.value_type
    numpy_type = constants.IBIS_TYPE_TO_PANDAS_TYPE.get(value_type, None)
    if numpy_type is None:
        raise ValueError(
            'Array value type must be a primitive type '
            '(e.g., number, string, or timestamp)'
        )
    return data.map(
        lambda array, numpy_type=numpy_type: [
            numpy_type(element) for element in array
        ]
    )


@execute_node.register(ops.Cast, pd.Series, dt.Timestamp)
def execute_cast_series_timestamp(op, data, type, **kwargs):
    arg = op.args[0]
    from_type = arg.type()

    if from_type.equals(type):  # noop cast
        return data

    tz = type.timezone

    if isinstance(from_type, (dt.Timestamp, dt.Date)):
        return data.astype(
            'M8[ns]' if tz is None else compat.DatetimeTZDtype('ns', tz)
        )

    if isinstance(from_type, (dt.String, dt.Integer)):
        timestamps = pd.to_datetime(
            data.values, infer_datetime_format=True, unit='ns',
        ).tz_localize(tz)
        return pd.Series(timestamps, index=data.index, name=data.name)

    raise TypeError("Don't know how to cast {} to {}".format(from_type, type))


def _normalize(values, original_index, name, timezone=None):
    index = pd.DatetimeIndex(values, tz=timezone)
    return pd.Series(index.normalize(), index=original_index, name=name)


@execute_node.register(ops.Cast, pd.Series, dt.Date)
def execute_cast_series_date(op, data, type, **kwargs):
    arg = op.args[0]
    from_type = arg.type()

    if from_type.equals(type):
        return data

    if isinstance(from_type, dt.Timestamp):
        return _normalize(
            data.values, data.index, data.name, timezone=from_type.timezone
        )

    if from_type.equals(dt.string):
        try:
            date_values = data.values.astype('datetime64[D]').astype(
                'datetime64[ns]'
            )
        except TypeError:
            date_values = _normalize(
                pd.to_datetime(
                    data.values, infer_datetime_format=True, box=False
                ),
                data.index,
                data.name,
            )
        return pd.Series(date_values, index=data.index, name=data.name)

    if isinstance(from_type, dt.Integer):
        return pd.Series(
            pd.to_datetime(data.values, box=False, unit='D'),
            index=data.index,
            name=data.name
        )

    raise TypeError("Don't know how to cast {} to {}".format(from_type, type))


@execute_node.register(ops.Negate, pd.Series)
def execute_series_unary_op_negate(op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    return np.negative(data)


@execute_node.register(ops.UnaryOp, pd.Series)
def execute_series_unary_op(op, data, **kwargs):
    function = getattr(np, type(op).__name__.lower())
    if data.dtype == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    return function(data)


def vectorize_object(op, arg, *args, **kwargs):
    func = np.vectorize(functools.partial(execute_node, op, **kwargs))
    return pd.Series(func(arg, *args), index=arg.index, name=arg.name)


@execute_node.register(
    ops.Log, pd.Series, (pd.Series, numbers.Real, decimal.Decimal, type(None))
)
def execute_series_log_with_base(op, data, base, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, base, **kwargs)

    if base is None:
        return np.log(data)
    return np.log(data) / np.log(base)


@execute_node.register(ops.Ln, pd.Series)
def execute_series_natural_log(op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    return np.log(data)


@execute_node.register(
    ops.Clip, pd.Series,
    (pd.Series, float, integer_types, type(None)),
    (pd.Series, float, integer_types, type(None))
)
def execute_series_clip(op, data, lower, upper, **kwargs):
    return data.clip(lower=lower, upper=upper)


@execute_node.register(
    ops.Quantile,
    (pd.Series, SeriesGroupBy), (float,) + six.integer_types
)
def execute_series_quantile(op, data, quantile, context=None, **kwargs):
    return context.agg(
        data, 'quantile', q=quantile, interpolation=op.interpolation
    )


@execute_node.register(ops.MultiQuantile, pd.Series, list)
def execute_series_quantile_list(op, data, quantile, context=None, **kwargs):
    result = context.agg(
        data, 'quantile', q=quantile, interpolation=op.interpolation
    )
    return list(result)


@execute_node.register(ops.Cast, datetime.datetime, dt.String)
def execute_cast_datetime_or_timestamp_to_string(op, data, type, **kwargs):
    """Cast timestamps to strings"""
    return str(data)


@execute_node.register(ops.Cast, datetime.datetime, dt.Int64)
def execute_cast_datetime_to_integer(op, data, type, **kwargs):
    """Cast datetimes to integers"""
    return pd.Timestamp(data).value


@execute_node.register(ops.Cast, pd.Timestamp, dt.Int64)
def execute_cast_timestamp_to_integer(op, data, type, **kwargs):
    """Cast timestamps to integers"""
    return data.value


@execute_node.register(
    ops.Cast,
    (np.bool_, bool),
    dt.Timestamp
)
def execute_cast_bool_to_timestamp(op, data, type, **kwargs):
    raise TypeError(
        'Casting boolean values to timestamps does not make sense. If you '
        'really want to cast boolean values to timestamps please cast to '
        'int64 first then to timestamp: '
        "value.cast('int64').cast('timestamp')"
    )


@execute_node.register(
    ops.Cast,
    integer_types + six.string_types,
    dt.Timestamp
)
def execute_cast_simple_literal_to_timestamp(op, data, type, **kwargs):
    """Cast integer and strings to timestamps"""
    return pd.Timestamp(data, tz=type.timezone)


@execute_node.register(ops.Cast, pd.Timestamp, dt.Timestamp)
def execute_cast_timestamp_to_timestamp(op, data, type, **kwargs):
    """Cast timestamps to other timestamps including timezone if necessary"""
    input_timezone = data.tz
    target_timezone = type.timezone

    if input_timezone == target_timezone:
        return data

    if input_timezone is None or target_timezone is None:
        return data.tz_localize(target_timezone)

    return data.tz_convert(target_timezone)


@execute_node.register(ops.Cast, datetime.datetime, dt.Timestamp)
def execute_cast_datetime_to_datetime(op, data, type, **kwargs):
    return execute_cast_timestamp_to_timestamp(
        op, data, type, **kwargs
    ).to_pydatetime()


@execute_node.register(
    ops.Cast, fixed_width_types + six.string_types, dt.DataType
)
def execute_cast_string_literal(op, data, type, **kwargs):
    try:
        cast_function = constants.IBIS_TO_PYTHON_LITERAL_TYPES[type]
    except KeyError:
        raise TypeError(
            "Don't know how to cast {!r} to type {}".format(data, type)
        )
    else:
        return cast_function(data)


@execute_node.register(
    ops.Round,
    scalar_types,
    (six.integer_types, type(None))
)
def execute_round_scalars(op, data, places, **kwargs):
    if places is None:
        return np.around(data)
    else:
        return np.around(data, places)


@execute_node.register(
    ops.Round,
    pd.Series,
    (pd.Series, np.integer, type(None)) + six.integer_types
)
def execute_round_series(op, data, places, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, places, **kwargs)
    return data.round(places if places is not None else 0)


@execute_node.register(ops.TableColumn, (pd.DataFrame, DataFrameGroupBy))
def execute_table_column_df_or_df_groupby(op, data, **kwargs):
    return data[op.name]


@execute_node.register(ops.Aggregation, pd.DataFrame)
def execute_aggregation_dataframe(op, data, scope=None, **kwargs):
    assert op.metrics, 'no metrics found during aggregation execution'

    if op.sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    predicates = op.predicates
    if predicates:
        predicate = functools.reduce(
            operator.and_,
            (execute(p, scope, **kwargs) for p in predicates)
        )
        data = data.loc[predicate]

    columns = {}

    if op.by:
        grouping_key_pairs = list(
            zip(op.by, map(operator.methodcaller('op'), op.by))
        )
        grouping_keys = [
            by_op.name if isinstance(by_op, ir.TableColumn)
            else execute(by, scope, **kwargs).rename(by.get_name())
            for by, by_op in grouping_key_pairs
        ]
        columns.update(
            (by_op.name, by.get_name()) for by, by_op in grouping_key_pairs
            if hasattr(by_op, 'name')
        )
        source = data.groupby(grouping_keys)
    else:
        source = data

    new_scope = toolz.merge(scope, {op.table.op(): source})
    pieces = [
        pd.Series(execute(metric, new_scope, **kwargs), name=metric.get_name())
        for metric in op.metrics
    ]

    result = pd.concat(pieces, axis=1).reset_index()
    result.columns = [columns.get(c, c) for c in result.columns]

    if op.having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not op.by:
            raise ValueError(
                'Filtering out aggregation values is not allowed without at '
                'least one grouping key'
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (execute(having, new_scope, **kwargs) for having in op.having)
        )
        assert len(predicate) == len(result), \
            'length of predicate does not match length of DataFrame'
        result = result.loc[predicate.values].reset_index(drop=True)
    return result


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(op, data, mask, context=None, **kwargs):
    return context.agg(data, type(op).__name__.lower())


variance_ddof = {
    'pop': 0,
    'sample': 1,
}


@execute_node.register(ops.Variance, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(op, data, _, context=None, **kwargs):
    return context.agg(data, 'var', ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(op, data, _, context=None, **kwargs):
    return context.agg(data, 'std', ddof=variance_ddof[op.how])


@execute_node.register(ops.CountDistinct, SeriesGroupBy, type(None))
def execute_count_distinct_series_groupby(op, data, _, context=None, **kwargs):
    return context.agg(data, 'nunique')


def _filtered_reduction(mask, method, data):
    return method(data[mask[data.index]])


@execute_node.register(ops.Reduction, SeriesGroupBy, SeriesGroupBy)
def execute_reduction_series_gb_mask(op, data, mask, context=None, **kwargs):
    method = operator.methodcaller(type(op).__name__.lower())
    return context.agg(
        data,
        functools.partial(_filtered_reduction, mask.obj, method)
    )


@execute_node.register(ops.CountDistinct, SeriesGroupBy, SeriesGroupBy)
def execute_count_distinct_series_groupby_mask(
    op, data, mask, context=None, **kwargs
):
    return context.agg(
        data,
        functools.partial(_filtered_reduction, mask.obj, pd.Series.nunique)
    )


@execute_node.register(ops.Variance, SeriesGroupBy, SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, context=None, **kwargs):
    return context.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].var(ddof=ddof)
        )
    )


@execute_node.register(ops.StandardDev, SeriesGroupBy, SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, context=None, **kwargs):
    return context.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].std(ddof=ddof)
        )
    )


@execute_node.register(ops.Count, DataFrameGroupBy, type(None))
def execute_count_frame_groupby(op, data, _, **kwargs):
    result = data.size()
    # FIXME(phillipc): We should not hard code this column name
    result.name = 'count'
    return result


@execute_node.register(ops.Reduction, pd.Series, (pd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, context=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return context.agg(operand, type(op).__name__.lower())


@execute_node.register(ops.CountDistinct, pd.Series, (pd.Series, type(None)))
def execute_count_distinct_series_mask(op, data, mask, context=None, **kwargs):
    return context.agg(data[mask] if mask is not None else data, 'nunique')


@execute_node.register(ops.StandardDev, pd.Series, (pd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, context=None, **kwargs):
    return context.agg(
        data[mask] if mask is not None else data,
        'std',
        ddof=variance_ddof[op.how]
    )


@execute_node.register(ops.Variance, pd.Series, (pd.Series, type(None)))
def execute_variance_series(op, data, mask, context=None, **kwargs):
    return context.agg(
        data[mask] if mask is not None else data,
        'var',
        ddof=variance_ddof[op.how]
    )


@execute_node.register((ops.Any, ops.All), pd.Series)
def execute_any_all_series(op, data, context=None, **kwargs):
    return context.agg(data, type(op).__name__.lower())


@execute_node.register(ops.NotAny, pd.Series)
def execute_notany_series(op, data, context=None, **kwargs):
    return ~context.agg(data, 'any')


@execute_node.register(ops.NotAll, pd.Series)
def execute_notall_series(op, data, context=None, **kwargs):
    return ~context.agg(data, 'all')


@execute_node.register(ops.Count, pd.DataFrame, type(None))
def execute_count_frame(op, data, _, **kwargs):
    return len(data)


@execute_node.register(ops.Not, (bool, np.bool_))
def execute_not_bool(op, data, **kwargs):
    return not data


@execute_node.register(
    ops.BinaryOp, (pd.Series, numeric_types), (pd.Series, numeric_types)
)
@execute_node.register(ops.StringConcat, pd.Series, pd.Series)
@execute_node.register(
    (ops.Comparison, ops.StringConcat), six.string_types, six.string_types
)
@execute_node.register(
    (ops.Comparison, ops.StringConcat, ops.Multiply),
    pd.Series, six.string_types
)
@execute_node.register(
    (ops.Comparison, ops.Multiply, ops.StringConcat),
    six.string_types, pd.Series
)
@execute_node.register(ops.Multiply, integer_types, six.string_types)
@execute_node.register(ops.Multiply, six.string_types, integer_types)
def execute_binary_op(op, left, right, **kwargs):
    op_type = type(op)
    try:
        operation = constants.BINARY_OPERATIONS[op_type]
    except KeyError:
        raise NotImplementedError(
            'Binary operation {} not implemented'.format(op_type.__name__)
        )
    else:
        return operation(left, right)


@execute_node.register(ops.BinaryOp, SeriesGroupBy, SeriesGroupBy)
def execute_binary_op_series_group_by(op, left, right, **kwargs):
    left_groupings = left.grouper.groupings
    right_groupings = right.grouper.groupings
    if left_groupings != right_groupings:
        raise ValueError(
            'Cannot perform {} operation on two series with '
            'different groupings'.format(type(op).__name__)
        )
    result = execute_node(op, left.obj, right.obj, **kwargs)
    return result.groupby(left_groupings)


@execute_node.register(ops.BinaryOp, SeriesGroupBy, simple_types)
def execute_binary_op_series_gb_simple(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register(ops.BinaryOp, simple_types, SeriesGroupBy)
def execute_binary_op_simple_series_gb(op, left, right, **kwargs):
    result = execute_node(op, left, right.obj, **kwargs)
    return result.groupby(right.grouper.groupings)


@execute_node.register(ops.UnaryOp, SeriesGroupBy)
def execute_unary_op_series_gb(op, operand, **kwargs):
    result = execute_node(op, operand.obj, **kwargs)
    return result.groupby(operand.grouper.groupings)


@execute_node.register(
    (ops.Log, ops.Round),
    SeriesGroupBy,
    (numbers.Real, decimal.Decimal, type(None))
)
def execute_log_series_gb_others(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register((ops.Log, ops.Round), SeriesGroupBy, SeriesGroupBy)
def execute_log_series_gb_series_gb(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right.obj, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register(ops.Not, pd.Series)
def execute_not_series(op, data, **kwargs):
    return ~data


@execute_node.register(ops.Strftime, pd.Timestamp, six.string_types)
def execute_strftime_timestamp_str(op, data, format_string, **kwargs):
    return data.strftime(format_string)


@execute_node.register(ops.Strftime, pd.Series, six.string_types)
def execute_strftime_series_str(op, data, format_string, **kwargs):
    return data.dt.strftime(format_string)


@execute_node.register(
    (ops.ExtractTimestampField, ops.ExtractTemporalField),
    pd.Timestamp
)
def execute_extract_timestamp_field_timestamp(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data, field_name)


@execute_node.register(ops.ExtractMillisecond, pd.Timestamp)
def execute_extract_millisecond_timestamp(op, data, **kwargs):
    return int(data.microsecond // 1000.0)


@execute_node.register(
    (ops.ExtractTimestampField, ops.ExtractTemporalField),
    pd.Series
)
def execute_extract_timestamp_field_series(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data.dt, field_name)


@execute_node.register(ops.NullIfZero, pd.Series)
def execute_null_if_zero_series(op, data, **kwargs):
    return data.where(data != 0, np.nan)


@execute_node.register(
    ops.StringSplit, pd.Series, (pd.Series,) + six.string_types
)
def execute_string_split(op, data, delimiter, **kwargs):
    return data.str.split(delimiter)


@execute_node.register(
    ops.Between,
    pd.Series,
    (pd.Series, numbers.Real, str, datetime.datetime),
    (pd.Series, numbers.Real, str, datetime.datetime)
)
def execute_between(op, data, lower, upper, **kwargs):
    return data.between(lower, upper)


@execute_node.register(
    ops.BetweenTime,
    pd.Series,
    (pd.Series, str, datetime.time),
    (pd.Series, str, datetime.time),
)
def execute_between_time(op, data, lower, upper, **kwargs):
    indexer = pd.DatetimeIndex(data).indexer_between_time(
        lower, upper)
    result = np.zeros(len(data), dtype=np.bool_)
    result[indexer] = True
    return result


@execute_node.register(ops.DistinctColumn, pd.Series)
def execute_series_distinct(op, data, **kwargs):
    return pd.Series(data.unique(), name=data.name)


@execute_node.register(ops.Union, pd.DataFrame, pd.DataFrame)
def execute_union_dataframe_dataframe(op, left, right, **kwargs):
    return pd.concat([left, right], axis=0)


@execute_node.register(ops.IsNull, pd.Series)
def execute_series_isnull(op, data, **kwargs):
    return data.isnull()


@execute_node.register(ops.NotNull, pd.Series)
def execute_series_notnnull(op, data, **kwargs):
    return data.notnull()


@execute_node.register(ops.SelfReference, pd.DataFrame)
def execute_node_self_reference_dataframe(op, data, **kwargs):
    return data


@execute_node.register(ir.ValueList)
def execute_node_value_list(op, **kwargs):
    return [execute(arg, **kwargs) for arg in op.values]


@execute_node.register(ops.Contains, pd.Series, list)
def execute_node_contains_series_list(op, data, elements, **kwargs):
    return data.isin(elements)


@execute_node.register(ops.NotContains, pd.Series, list)
def execute_node_not_contains_series_list(op, data, elements, **kwargs):
    return ~data.isin(elements)


@execute_node.register(ops.Where,
                       pd.Series,
                       (pd.Series,) + scalar_types,
                       (pd.Series,) + scalar_types)
def execute_node_where_series(op, cond, true, false, **kwargs):
    if isinstance(true, scalar_types):
        true = pd.Series(np.repeat(true, len(cond)))
    # No need to turn false into a series, pandas will broadcast it
    return true.where(cond, other=false)


@execute_node.register(ops.Where,
                       scalar_types,
                       (pd.Series,) + scalar_types,
                       (pd.Series,) + scalar_types)
def execute_node_where_scalar(op, cond, true, false, **kwargs):
    # Note that it is not necessary to check that true and false are also
    # scalars. This allows users to do things like:
    # ibis.where(even_or_odd_bool, [2, 4, 6], [1, 3, 5])
    return true if cond else false
