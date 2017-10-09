from __future__ import absolute_import

import numbers
import operator
import datetime
import functools
import decimal
import collections
import re

import six

import numpy as np
import pandas as pd

from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy

import toolz

from ibis import compat

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis import util

import ibis.pandas.context as ctx
from ibis.pandas.core import (
    integer_types, simple_types, numeric_types, fixed_width_types
)
from ibis.pandas.dispatch import execute, execute_node, execute_first


@execute_node.register(ir.Literal)
@execute_node.register(ir.Literal, object)
@execute_node.register(ir.Literal, object, dt.DataType)
def execute_node_literal(op, *args, **kwargs):
    return op.value


_IBIS_TYPE_TO_PANDAS_TYPE = {
    dt.float: np.float32,
    dt.double: np.float64,
    dt.int8: np.int8,
    dt.int16: np.int16,
    dt.int32: np.int32,
    dt.int64: np.int64,
    dt.string: str,
    dt.timestamp: 'datetime64[ns]',
}


@execute_node.register(ops.Limit, pd.DataFrame, integer_types, integer_types)
def execute_limit_frame(op, data, limit, offset, **kwargs):
    return data.iloc[offset:offset + limit]


@execute_node.register(ops.Cast, pd.Series, dt.DataType)
def execute_cast_series_generic(op, data, type, **kwargs):
    return data.astype(_IBIS_TYPE_TO_PANDAS_TYPE[type])


@execute_node.register(ops.Cast, pd.Series, dt.Array)
def execute_cast_series_array(op, data, type, **kwargs):
    value_type = type.value_type
    numpy_type = _IBIS_TYPE_TO_PANDAS_TYPE.get(value_type, None)
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


_LITERAL_CAST_TYPES = {
    dt.double: float,
    dt.float: float,
    dt.int64: int,
    dt.int32: int,
    dt.int16: int,
    dt.int8: int,
    dt.string: str,
    dt.date: lambda x: pd.Timestamp(x).to_pydatetime().date(),
}


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
        cast_function = _LITERAL_CAST_TYPES[type]
    except KeyError:
        raise TypeError(
            "Don't know how to cast {!r} to type {}".format(data, type)
        )
    else:
        return cast_function(data)


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


def _compute_sort_key(key, data, **kwargs):
    by = key.args[0]
    try:
        return by.get_name(), None
    except com.ExpressionError:
        name = util.guid()
        new_scope = {t: data for t in by.op().root_tables()}
        new_column = execute(by, new_scope, **kwargs)
        new_column.name = name
        return name, new_column


def _compute_sorted_frame(sort_keys, df, **kwargs):
    computed_sort_keys = []
    ascending = [key.op().ascending for key in sort_keys]
    new_columns = {}

    for i, key in enumerate(map(operator.methodcaller('op'), sort_keys)):
        computed_sort_key, temporary_column = _compute_sort_key(
            key, df, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)

        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(computed_sort_keys, ascending=ascending)
    result = result.drop(new_columns.keys(), axis=1)
    return result


@execute_node.register(ops.Selection, pd.DataFrame)
def execute_selection_dataframe(op, data, scope=None, **kwargs):
    selections = op.selections
    predicates = op.predicates
    sort_keys = op.sort_keys

    result = data

    if selections:
        data_pieces = []
        for selection in selections:
            table_op = op.table.op()
            selection_operation = selection.op()

            if op.table is selection:
                pandas_object = data
            elif isinstance(selection, ir.ScalarExpr):
                root_tables = selection_operation.root_tables()
                additional_scope = collections.OrderedDict(
                    zip(root_tables, (data for _ in range(len(root_tables))))
                )
                new_scope = toolz.merge(
                    scope,
                    additional_scope,
                    factory=collections.OrderedDict,
                )
                pandas_object = execute(selection, new_scope, **kwargs)
            elif isinstance(selection, ir.ColumnExpr):
                if isinstance(selection_operation, ir.TableColumn):
                    # slightly faster path for simple column selection
                    pandas_object = data[selection_operation.name]
                elif isinstance(table_op, ops.Join):
                    pandas_object = execute(
                        selection,
                        toolz.merge(
                            scope, {selection_operation.table.op(): data}
                        ),
                        **kwargs
                    )
                else:
                    pandas_object = execute(
                        selection,
                        toolz.merge(scope, {op.table.op(): data}),
                        **kwargs
                    )
            elif isinstance(selection, ir.TableExpr):
                # These two statements should never raise unless our
                # assumptions are wrong because:
                # 1. If we're selecting ourself, then we've already caught that
                #    case above
                # 2. We've checked that `s` originates from its parent before
                #    executing
                assert isinstance(table_op, ops.Join)
                assert selection.equals(table_op.left) or selection.equals(
                    table_op.right
                )
                pandas_object = data[selection.columns]
            else:
                raise TypeError(
                    "Don't know how to compute selection of type {}".format(
                        type(selection_operation).__name__
                    )
                )

            if isinstance(pandas_object, pd.Series):
                pandas_object = pandas_object.rename(
                    getattr(selection, '_name', pandas_object.name)
                )
            data_pieces.append(pandas_object)
        result = pd.concat(data_pieces, axis=1)

    if predicates:
        where = functools.reduce(
            operator.and_, (execute(p, scope, **kwargs) for p in predicates)
        )
        result = result.loc[where]

    if sort_keys:
        result = _compute_sorted_frame(sort_keys, result, **kwargs)
    return result.reset_index(drop=True)


@execute_node.register(ops.Aggregation, pd.DataFrame)
def execute_aggregation_dataframe(op, data, scope=None, **kwargs):
    assert op.metrics

    if op.having:
        raise NotImplementedError('having expressions not yet implemented')

    if op.sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    predicates = op.predicates
    if predicates:
        predicate = functools.reduce(
            operator.and_, (execute(p, scope, **kwargs) for p in predicates)
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

    df = pd.concat(pieces, axis=1).reset_index()
    df.columns = [columns.get(c, c) for c in df.columns]
    return df


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(op, data, mask, context=None, **kwargs):
    return context.agg(data, type(op).__name__.lower())


@execute_node.register(ops.Variance, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(op, data, _, context=None, **kwargs):
    return context.agg(data, 'var')


@execute_node.register(ops.StandardDev, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(op, data, _, context=None, **kwargs):
    return context.agg(data, 'std')


@execute_node.register(ops.CountDistinct, SeriesGroupBy)
def execute_count_distinct_series_groupby(op, data, context=None, **kwargs):
    return context.agg(data, 'nunique')


@execute_node.register(ops.Reduction, SeriesGroupBy, SeriesGroupBy)
def execute_reduction_series_gb_mask(op, data, mask, context=None, **kwargs):
    method = operator.methodcaller(type(op).__name__.lower())
    return context.agg(
        data,
        lambda x, mask=mask.obj, method=method: method(x[mask[x.index]])
    )


@execute_node.register(ops.Variance, SeriesGroupBy, SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, context=None, **kwargs):
    return context.agg(data, lambda x, mask=mask.obj: x[mask[x.index]].var())


@execute_node.register(ops.StandardDev, SeriesGroupBy, SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, context=None, **kwargs):
    return context.agg(data, lambda x, mask=mask.obj: x[mask[x.index]].std())


@execute_node.register(ops.GroupConcat, SeriesGroupBy, six.string_types)
def execute_group_concat_series_gb(op, data, sep, context=None, **kwargs):
    return context.agg(data, lambda x, sep=sep: sep.join(x.astype(str)))


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


@execute_node.register(ops.StandardDev, pd.Series, (pd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, context=None, **kwargs):
    return context.agg(data[mask] if mask is not None else data, 'std')


@execute_node.register(ops.Variance, pd.Series, (pd.Series, type(None)))
def execute_variance_series(op, data, mask, context=None, **kwargs):
    return context.agg(data[mask] if mask is not None else data, 'var')


@execute_node.register(
    ops.GroupConcat,
    pd.Series, six.string_types, (pd.Series, type(None))
)
def execute_group_concat_series_mask(op, data, sep, mask, **kwargs):
    return sep.join(data[mask] if mask is not None else data)


@execute_node.register(ops.GroupConcat, pd.Series, six.string_types)
def execute_group_concat_series(op, data, sep, **kwargs):
    return sep.join(data.astype(str))


@execute_node.register((ops.Any, ops.All), pd.Series)
def execute_any_all_series(op, data, context=None, **kwargs):
    return context.agg(data, type(op).__name__.lower())


@execute_node.register(ops.CountDistinct, pd.Series)
def execute_count_distinct_series(op, data, **kwargs):
    # TODO(phillipc): Does count distinct have a mask?
    return data.nunique()


@execute_node.register(ops.Count, pd.DataFrame, type(None))
def execute_count_frame(op, data, _, **kwargs):
    return len(data)


@execute_node.register(ops.Not, (bool, np.bool_))
def execute_not_bool(op, data, **kwargs):
    return not data


_JOIN_TYPES = {
    ops.LeftJoin: 'left',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


@execute_node.register(ops.Join, pd.DataFrame, pd.DataFrame)
def execute_materialized_join(op, left, right, **kwargs):
    try:
        how = _JOIN_TYPES[type(op)]
    except KeyError:
        raise NotImplementedError('{} not supported'.format(type(op).__name__))

    overlapping_columns = set(left.columns) & set(right.columns)

    left_on, right_on = _extract_predicate_names(op.predicates)
    _validate_columns(overlapping_columns, left_on, right_on)

    return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on)


@execute_node.register(ops.AsOfJoin, pd.DataFrame, pd.DataFrame)
def execute_asof_join(op, left, right, **kwargs):
    overlapping_columns = set(left.columns) & set(right.columns)
    left_on, right_on = _extract_predicate_names(op.predicates)
    left_by, right_by = _extract_predicate_names(op.by_predicates)
    _validate_columns(
        overlapping_columns, left_on, right_on, left_by, right_by)

    return pd.merge_asof(
        left=left,
        right=right,
        left_on=left_on,
        right_on=right_on,
        left_by=left_by or None,
        right_by=right_by or None
    )


def _extract_predicate_names(predicates):
    lefts = []
    rights = []
    for predicate in map(operator.methodcaller('op'), predicates):
        if not isinstance(predicate, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with pandas'
            )
        left_name = predicate.left._name
        right_name = predicate.right._name
        lefts.append(left_name)
        rights.append(right_name)
    return lefts, rights


def _validate_columns(orig_columns, *key_lists):
    overlapping_columns = orig_columns.difference(
        item for sublist in key_lists for item in sublist
    )
    if overlapping_columns:
        raise ValueError(
            'left and right DataFrame columns overlap on {} in a join. '
            'Please specify the columns you want to select from the join, '
            'e.g., join[left.column1, right.column2, ...]'.format(
                overlapping_columns
            )
        )


_BINARY_OPERATIONS = {
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
}


@execute_node.register(ops.BinaryOp, pd.Series, (pd.Series,) + simple_types)
@execute_node.register(ops.BinaryOp, numeric_types, numeric_types)
@execute_node.register(ops.BinaryOp, six.string_types, six.string_types)
def execute_binary_op(op, left, right, **kwargs):
    op_type = type(op)
    try:
        operation = _BINARY_OPERATIONS[op_type]
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
    result = execute_binary_op(op, left.obj, right.obj)
    return result.groupby(left_groupings)


@execute_node.register(ops.BinaryOp, SeriesGroupBy, simple_types)
def execute_binary_op_series_gb(op, left, right, **kwargs):
    result = execute_binary_op(op, left.obj, right)
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


@execute_node.register(ops.StringLength, pd.Series)
def execute_string_length_series(op, data, **kwargs):
    return data.str.len()


@execute_node.register(
    ops.Substring,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + integer_types
)
def execute_string_substring(op, data, start, length, **kwargs):
    return data.str[start:start + length]


@execute_node.register(ops.Strip, pd.Series)
def execute_string_strip(op, data, **kwargs):
    return data.str.strip()


@execute_node.register(ops.LStrip, pd.Series)
def execute_string_lstrip(op, data, **kwargs):
    return data.str.lstrip()


@execute_node.register(ops.RStrip, pd.Series)
def execute_string_rstrip(op, data, **kwargs):
    return data.str.rstrip()


@execute_node.register(
    ops.LPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_lpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='left', fillchar=pad)


@execute_node.register(
    ops.RPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_rpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='right', fillchar=pad)


@execute_node.register(ops.Reverse, pd.Series)
def execute_string_reverse(op, data, **kwargs):
    return data.str[::-1]


@execute_node.register(ops.Lowercase, pd.Series)
def execute_string_lower(op, data, **kwargs):
    return data.str.lower()


@execute_node.register(ops.Uppercase, pd.Series)
def execute_string_upper(op, data, **kwargs):
    return data.str.upper()


@execute_node.register(ops.Capitalize, pd.Series)
def execute_string_capitalize(op, data, **kwargs):
    return data.str.capitalize()


@execute_node.register(ops.Repeat, pd.Series, (pd.Series,) + integer_types)
def execute_string_repeat(op, data, times, **kwargs):
    return data.str.repeat(times)


@execute_node.register(
    ops.StringFind,
    pd.Series,
    (pd.Series,) + six.string_types,
    (pd.Series, type(None)) + integer_types,
    (pd.Series, type(None)) + integer_types,
)
def execute_string_contains(op, data, needle, start, end, **kwargs):
    return data.str.find(needle, start, end)


@execute_node.register(
    ops.StringSQLLike,
    pd.Series,
    (pd.Series,) + six.string_types,
)
def execute_string_like(op, data, pattern, **kwargs):
    return data.str.contains(pattern, regex=True)


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


@execute_node.register(ops.ArrayLength, pd.Series)
def execute_array_length(op, data, **kwargs):
    return data.apply(len)


@execute_node.register(ops.ArrayLength, list)
def execute_array_length_scalar(op, data, **kwargs):
    return len(data)


@execute_node.register(
    ops.ArraySlice,
    pd.Series, six.integer_types, (six.integer_types, type(None))
)
def execute_array_slice(op, data, start, stop, **kwargs):
    return data.apply(operator.itemgetter(slice(start, stop)))


@execute_node.register(
    ops.ArraySlice,
    list, six.integer_types, (six.integer_types, type(None))
)
def execute_array_slice_scalar(op, data, start, stop, **kwargs):
    return data[start:stop]


@execute_node.register(ops.ArrayIndex, pd.Series, six.integer_types)
def execute_array_index(op, data, index, **kwargs):
    return data.apply(
        lambda array, index=index: (
            array[index] if -len(array) <= index < len(array) else None
        )
    )


@execute_node.register(ops.ArrayIndex, list, six.integer_types)
def execute_array_index_scalar(op, data, index, **kwargs):
    try:
        return data[index]
    except IndexError:
        return None


@execute_node.register(ops.ArrayConcat, pd.Series, (pd.Series, list))
@execute_node.register(ops.ArrayConcat, list, pd.Series)
@execute_node.register(ops.ArrayConcat, list, list)
def execute_array_concat(op, left, right, **kwargs):
    return left + right


@execute_node.register(ops.ArrayRepeat, pd.Series, pd.Series)
@execute_node.register(ops.ArrayRepeat, six.integer_types, (pd.Series, list))
@execute_node.register(ops.ArrayRepeat, (pd.Series, list), six.integer_types)
def execute_array_repeat(op, left, right, **kwargs):
    return left * right


@execute_node.register(ops.ArrayCollect, pd.Series)
def execute_array_collect(op, data, **kwargs):
    return list(data)


@execute_node.register(ops.ArrayCollect, SeriesGroupBy)
def execute_array_collect_group_by(op, data, **kwargs):
    return data.apply(list)


@execute_node.register(ops.CumulativeSum, pd.Series)
def execute_series_cumsum(op, data, **kwargs):
    return data.cumsum()


@execute_node.register(ops.CumulativeMin, pd.Series)
def execute_series_cummin(op, data, **kwargs):
    return data.cummin()


@execute_node.register(ops.CumulativeMax, pd.Series)
def execute_series_cummax(op, data, **kwargs):
    return data.cummax()


@execute_node.register(ops.CumulativeOp, pd.Series)
def execute_series_cumulative_op(op, data, **kwargs):
    typename = type(op).__name__
    match = re.match(r'^Cumulative([A-Za-z_][A-Za-z0-9_]*)$', typename)
    if match is None:
        raise ValueError('Unknown operation {}'.format(typename))

    try:
        operation_name, = match.groups()
    except ValueError:
        raise ValueError(
            'More than one operation name found in {} class'.format(typename)
        )
    return ctx.Cumulative().agg(data, operation_name.lower())


@execute_node.register(
    ops.Lag, (pd.Series, SeriesGroupBy),
    integer_types + (type(None),),
    type(None),
)
def execute_series_lag(op, data, offset, default, **kwargs):
    return data.shift(1 if offset is None else offset)


@execute_node.register(
    ops.Lead, (pd.Series, SeriesGroupBy),
    integer_types + (type(None),),
    type(None),
)
def execute_series_lead(op, data, offset, default, **kwargs):
    return data.shift(-(1 if offset is None else offset))


@execute_node.register(ops.FirstValue, pd.Series)
def execute_series_first_value(op, data, **kwargs):
    return data.iloc[0]


@execute_node.register(ops.FirstValue, SeriesGroupBy)
def execute_series_group_by_first_value(op, data, context=None, **kwargs):
    return context.agg(data, 'first')


@execute_node.register(ops.LastValue, pd.Series)
def execute_series_last_value(op, data, **kwargs):
    return data.iloc[-1]


@execute_node.register(ops.LastValue, SeriesGroupBy)
def execute_series_group_by_last_value(op, data, context=None, **kwargs):
    return context.agg(data, 'last')


@execute_node.register(ops.MinRank, (pd.Series, SeriesGroupBy))
def execute_series_min_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='min', ascending=True)


@execute_node.register(ops.DenseRank, (pd.Series, SeriesGroupBy))
def execute_series_dense_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='dense', ascending=True)


@execute_node.register(ops.PercentRank, (pd.Series, SeriesGroupBy))
def execute_series_percent_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='min', ascending=True, pct=True)


def _post_process_empty(scalar, index):
    return pd.Series(scalar, index=index)


def _post_process_group_by(series, index):
    return series


def _post_process_order_by(series, index):
    return series.reindex(index)


def _post_process_group_by_order_by(series, index):
    level_list = list(range(series.index.nlevels - 1))
    series_with_reset_index = series.reset_index(level=level_list, drop=True)
    reindexed_series = series_with_reset_index.reindex(index)
    return reindexed_series


@execute_first.register(ops.WindowOp, pd.DataFrame)
def execute_frame_window_op(op, data, scope=None, context=None, **kwargs):
    operand, window = op.args

    following = window.following
    order_by = window._order_by

    if order_by and following != 0:
        raise ValueError(
            'Following with a value other than 0 (current row) with order_by '
            'is not yet implemented in the pandas backend. Use '
            'ibis.trailing_window or ibis.cumulative_window to '
            'construct windows when using pandas.'
        )

    group_by = window._group_by
    grouping_keys = [
        key_op.name if isinstance(key_op, ir.TableColumn) else execute(
            key,
            context=context,
            **kwargs
        )
        for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by

    if grouping_keys:
        source = data.groupby(grouping_keys, sort=False, as_index=not order_by)

        if order_by:
            sorted_df = source.apply(
                lambda df, order_by=order_by, kwargs=kwargs: (
                    _compute_sorted_frame(order_by, df, **kwargs)
                )
            )
            source = sorted_df.groupby(grouping_keys, sort=False)
            post_process = _post_process_group_by_order_by
        else:
            post_process = _post_process_group_by
    else:
        if order_by:
            source = _compute_sorted_frame(order_by, data, **kwargs)
            post_process = _post_process_order_by
        else:
            source = data
            post_process = _post_process_empty

    new_scope = toolz.merge(
        scope,
        collections.OrderedDict(
            (t, source) for t in operand.op().root_tables()
        ),
        factory=collections.OrderedDict,
    )

    # no order by or group by: default summarization context
    #
    # if we're reducing and we have an order by expression then we need to
    # expand or roll.
    #
    # otherwise we're transforming
    if not grouping_keys and not order_by:
        context = ctx.Summarize()
    elif isinstance(operand.op(), ops.Reduction) and order_by:
        preceding = window.preceding
        if preceding is not None:
            context = ctx.Trailing(preceding)
        else:
            context = ctx.Cumulative()
    else:
        context = ctx.Transform()

    result = execute(operand, new_scope, context=context, **kwargs)
    return post_process(result, data.index)
