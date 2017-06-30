import numbers
import operator
import datetime
import functools

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

from ibis.pandas.core import (
    integer_types, simple_types, numeric_types, fixed_width_types
)
from ibis.pandas.dispatch import execute, execute_node


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
def execute_limit_frame(op, data, limit, offset, scope=None):
    return data.iloc[offset:offset + limit]


@execute_node.register(ops.Cast, pd.Series, dt.DataType)
def execute_cast_series_generic(op, data, type, scope=None):
    return data.astype(_IBIS_TYPE_TO_PANDAS_TYPE[type])


@execute_node.register(ops.Cast, pd.Series, dt.Timestamp)
def execute_cast_series_timestamp(op, data, type, scope=None):
    # TODO(phillipc): Consistent units
    tz = type.timezone
    return data.astype(
        'datetime64[ns]' if tz is None else compat.DatetimeTZDtype('ns', tz)
    )


@execute_node.register(ops.Cast, pd.Series, dt.Date)
def execute_cast_series_date(op, data, _, scope=None):
    return data.dt.date.astype('datetime64[D]')


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


def frame_chunks(df):
    if isinstance(df, DataFrameGroupBy):
        for name, chunk in df:
            yield name, chunk
    else:
        yield None, df


@execute_node.register(ops.Cast, datetime.datetime, dt.String)
def execute_cast_datetime_or_timestamp_to_string(op, data, type, scope=None):
    """Cast timestamps to strings"""
    return str(data)


@execute_node.register(ops.Cast, datetime.datetime, dt.Int64)
def execute_cast_datetime_to_integer(op, data, type, scope=None):
    """Cast datetimes to integers"""
    return pd.Timestamp(data).value


@execute_node.register(ops.Cast, pd.Timestamp, dt.Int64)
def execute_cast_timestamp_to_integer(op, data, type, scope=None):
    """Cast timestamps to integers"""
    return data.value


@execute_node.register(
    ops.Cast,
    (np.bool_, bool),
    dt.Timestamp
)
def execute_cast_bool_to_timestamp(op, data, type, scope=None):
    raise TypeError(
        'Casting boolean values to timestamps does not make sense. If you '
        'really want to cast boolean values to timestamps please cast to '
        'int64 first then to timestamp: '
        "value.cast('int64').cast('timestamp')"
    )


@execute_node.register(
    ops.Cast,
    six.integer_types + six.string_types,
    dt.Timestamp
)
def execute_cast_simple_literal_to_timestamp(op, data, type, scope=None):
    """Cast integer and strings to timestamps"""
    return pd.Timestamp(data, tz=type.timezone)


@execute_node.register(ops.Cast, pd.Timestamp, dt.Timestamp)
def execute_cast_timestamp_to_timestamp(op, data, type, scope=None):
    """Cast timestamps to other timestamps including timezone if necessary"""
    input_timezone = data.tz
    target_timezone = type.timezone

    if input_timezone == target_timezone:
        return data

    if input_timezone is None or target_timezone is None:
        return data.tz_localize(target_timezone)

    return data.tz_convert(target_timezone)


@execute_node.register(ops.Cast, datetime.datetime, dt.Timestamp)
def execute_cast_datetime_to_datetime(op, data, type, scope=None):
    return execute_cast_timestamp_to_timestamp(
        op, data, type, scope=scope
    ).to_pydatetime()


@execute_node.register(
    ops.Cast, fixed_width_types + six.string_types, dt.DataType
)
def execute_cast_string_literal(op, data, type, scope=None):
    try:
        cast_function = _LITERAL_CAST_TYPES[type]
    except KeyError:
        raise TypeError(
            "Don't know how to cast {!r} to type {}".format(data, type)
        )
    else:
        return cast_function(data)


@execute_node.register(ops.TableColumn, (pd.DataFrame, DataFrameGroupBy))
def execute_table_column_dataframe_or_dataframe_groupby(op, data, scope=None):
    return data[op.name]


def _compute_sort_key(key, scope):
    by = key.args[0]
    try:
        return by.get_name(), None
    except com.ExpressionError:
        name = util.guid()
        new_column = execute(by, scope)
        new_column.name = name
        return name, new_column


@execute_node.register(ops.Selection, pd.DataFrame)
def execute_selection_dataframe(op, data, scope=None):
    selections = op.selections
    predicates = op.predicates
    sort_keys = op.sort_keys

    result = data

    if selections:
        data_pieces = []
        for s in selections:
            if op.table is s:
                pandas_object = data
            elif isinstance(s, ir.ColumnExpr):
                if isinstance(op.table.op(), ops.Join):
                    pandas_object = execute(
                        s, toolz.merge(scope, {s.op().table: data})
                    )
                else:
                    pandas_object = execute(
                        s, toolz.merge(scope, {op.table: data})
                    )
            else:
                raise TypeError(
                    "Don't know how to compute selection of type {}".format(
                        type(s).__name__
                    )
                )

            if isinstance(pandas_object, pd.Series):
                pandas_object = pandas_object.rename(
                    getattr(s, '_name', pandas_object.name)
                )
            data_pieces.append(pandas_object)
        result = pd.concat(data_pieces, axis=1)

    if predicates:
        where = functools.reduce(
            operator.and_, (execute(p, scope) for p in predicates)
        )
        result = result.loc[where]

    column_names = list(result.columns)

    if sort_keys:

        computed_sort_keys = [None] * len(sort_keys)
        ascending = [key.op().ascending for key in sort_keys]
        temporary_columns = [None] * len(sort_keys)

        for i, key in enumerate(map(operator.methodcaller('op'), sort_keys)):
            computed_sort_keys[i], temporary_columns[i] = _compute_sort_key(
                key, {op.table: result}
            )

        if temporary_columns:
            result = pd.concat(
                [result] + [c for c in temporary_columns if c is not None],
                axis=1
            )
        result = result.sort_values(computed_sort_keys, ascending=ascending)
    return result.loc[:, column_names]


@execute_node.register(ops.Aggregation, pd.DataFrame)
def execute_aggregation_dataframe(op, data, scope=None):
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
            operator.and_, (execute(p, scope) for p in predicates)
        )
        data = data.loc[predicate]

    if op.by:
        source = data.groupby([execute(by, scope) for by in op.by])
    else:
        source = data

    metrics = op.metrics
    ops = [metric.op() for metric in metrics]
    metric_names = [metric.get_name() for metric in metrics]
    first_args = [metric_op.args[0] for metric_op in ops]
    first_arg_names = []
    for arg in first_args:
        try:
            first_arg_name = arg.get_name()
        except (AttributeError, com.ExpressionError):
            first_arg_name = None
        first_arg_names.append(first_arg_name)

    pieces = []

    index_name = [b.get_name() for b in op.by] if op.by else [None]

    for metric, first_arg, first_arg_name in zip(
        metrics, first_args, first_arg_names
    ):
        pairs = [
            key if isinstance(key, tuple) else (key,)
            for key, _ in frame_chunks(source)
        ]
        index = pd.MultiIndex.from_tuples(pairs, names=index_name)
        piece = pd.Series(
            [
                execute(
                    metric,
                    toolz.merge(scope, {op.table: chunk})
                ) for _, chunk in frame_chunks(source)
            ],
            name=metric.get_name(),
            index=index
        )
        pieces.append(piece)

    data_pieces = [
        p if isinstance(p, (pd.Series, pd.DataFrame))
        else pd.Series(p, name=name)
        for p, name in zip(pieces, first_arg_names)
    ]

    return pd.concat(data_pieces, axis=1).rename(
        columns=dict(zip(first_arg_names, metric_names))
    ).reset_index()


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(op, data, mask, scope=None):
    return getattr(data, type(op).__name__.lower())()


@execute_node.register(ops.CountDistinct, SeriesGroupBy)
def execute_count_distinct_series_groupby(op, data, scope=None):
    return data.nunique()


@execute_node.register(ops.Reduction, SeriesGroupBy, pd.Series)
def execute_reduction_series_groupby_mask(op, data, mask, scope=None):
    method = operator.methodcaller(type(op).__name__.lower())
    return data.apply(
        lambda x, mask=mask, method=method: method(x[mask[x.index]])
    )


@execute_node.register(ops.GroupConcat, SeriesGroupBy, six.string_types)
def execute_group_concat_series_groupby(op, data, sep, scope=None):
    return data.apply(lambda x, sep=sep: sep.join(x.astype(str)))


@execute_node.register(ops.Count, DataFrameGroupBy, type(None))
def execute_count_frame_groupby(op, data, _, scope=None):
    result = data.size()
    # FIXME(phillipc): We should not hard code this column name
    result.name = 'count'
    return result


@execute_node.register(ops.Reduction, pd.Series, (pd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, scope=None):
    operand = data[mask] if mask is not None else data
    return getattr(operand, type(op).__name__.lower())()


@execute_node.register(ops.StandardDev, pd.Series, (pd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, scope=None):
    return (data[mask] if mask is not None else data).std()


@execute_node.register(ops.Variance, pd.Series, (pd.Series, type(None)))
def execute_variance_series(op, data, mask, scope=None):
    return (data[mask] if mask is not None else data).var()


@execute_node.register(
    ops.GroupConcat,
    pd.Series, six.string_types, (pd.Series, type(None))
)
def execute_group_concat_series_mask(op, data, sep, mask, scope=None):
    return sep.join(data[mask] if mask is not None else data)


@execute_node.register(ops.GroupConcat, pd.Series, six.string_types)
def execute_group_concat_series(op, data, sep, scope=None):
    return sep.join(data.astype(str))


@execute_node.register((ops.Any, ops.All), pd.Series)
def execute_any_all_series(op, data, scope=None):
    return getattr(data, type(op).__name__.lower())()


@execute_node.register(ops.CountDistinct, pd.Series)
def execute_count_distinct_series(op, data, scope=None):
    # TODO(phillipc): Does count distinct have a mask?
    return data.nunique()


@execute_node.register(ops.Count, pd.DataFrame, type(None))
def execute_count_frame(op, data, _, scope=None):
    return len(data)


@execute_node.register(ops.Not, (bool, np.bool_))
def execute_not_bool(op, data, scope=None):
    return not data


_JOIN_TYPES = {
    ops.LeftJoin: 'left',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


@execute_node.register(ops.Join, pd.DataFrame, pd.DataFrame)
def execute_materialized_join(op, left, right, scope=None):
    try:
        how = _JOIN_TYPES[type(op)]
    except KeyError:
        raise NotImplementedError('{} not supported'.format(type(op).__name__))

    overlapping_columns = set(left.columns) & set(right.columns)

    left_on = []
    right_on = []

    for predicate in map(operator.methodcaller('op'), op.predicates):
        if not isinstance(predicate, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with pandas'
            )
        left_name = predicate.left._name
        right_name = predicate.right._name
        left_on.append(left_name)
        right_on.append(right_name)

        # TODO(phillipc): Is this the correct approach? That is, can we safely
        #                 ignore duplicate join keys?
        overlapping_columns -= {left_name, right_name}

    if overlapping_columns:
        raise ValueError(
            'left and right DataFrame columns overlap on {} in a join. '
            'Please specify the columns you want to select from the join, '
            'e.g., join[left.column1, right.column2, ...]'.format(
                overlapping_columns
            )
        )

    return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on)


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


@execute_node.register(ops.BinaryOp, (pd.Series,), (pd.Series,) + simple_types)
@execute_node.register(ops.BinaryOp, numeric_types, numeric_types)
@execute_node.register(ops.BinaryOp, six.string_types, six.string_types)
def execute_binary_operation_series(op, left, right, scope=None):
    op_type = type(op)
    try:
        operation = _BINARY_OPERATIONS[op_type]
    except KeyError:
        raise NotImplementedError(
            'Binary operation {} not implemented'.format(op_type.__name__)
        )
    else:
        return operation(left, right)


@execute_node.register(ops.Comparison, SeriesGroupBy, SeriesGroupBy)
def execute_binary_operation_series_group_by(op, left, right, scope=None):
    return execute_binary_operation_series(op, left.obj, right.obj)


@execute_node.register(ops.Comparison, SeriesGroupBy, simple_types)
def execute_binary_operation_series_group_by_scalar(
    op, left, right, scope=None
):
    return execute_binary_operation_series(op, left.obj, right)


@execute_node.register(ops.Not, pd.Series)
def execute_not_series(op, data, scope=None):
    return ~data


@execute_node.register(ops.Strftime, pd.Timestamp, six.string_types)
def execute_strftime_timestamp_str(op, data, format_string, scope=None):
    return data.strftime(format_string)


@execute_node.register(ops.Strftime, pd.Series, six.string_types)
def execute_strftime_series_str(op, data, format_string, scope=None):
    return data.dt.strftime(format_string)


@execute_node.register(
    (ops.ExtractTimestampField, ops.ExtractTemporalField),
    pd.Timestamp
)
def execute_extract_timestamp_field_timestamp(op, data, scope=None):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data, field_name)


@execute_node.register(ops.ExtractMillisecond, pd.Timestamp)
def execute_extract_millisecond_timestamp(op, data, scope=None):
    return int(data.microsecond // 1000.0)


@execute_node.register(
    (ops.ExtractTimestampField, ops.ExtractTemporalField),
    pd.Series
)
def execute_extract_timestamp_field_series(op, data, scope=None):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data.dt, field_name)


@execute_node.register(ops.NullIfZero, pd.Series)
def execute_null_if_zero_series(op, data, scope=None):
    return data.where(data != 0, np.nan)


@execute_node.register(ops.StringLength, pd.Series)
def execute_string_length_series(op, data, scope=None):
    return data.str.len()


@execute_node.register(
    ops.Substring,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + integer_types
)
def execute_string_substring(op, data, start, length, scope=None):
    return data.str[start:start + length]


@execute_node.register(ops.Strip, pd.Series)
def execute_string_strip(op, data, scope=None):
    return data.str.strip()


@execute_node.register(ops.LStrip, pd.Series)
def execute_string_lstrip(op, data, scope=None):
    return data.str.lstrip()


@execute_node.register(ops.RStrip, pd.Series)
def execute_string_rstrip(op, data, scope=None):
    return data.str.rstrip()


@execute_node.register(
    ops.LPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_lpad(op, data, length, pad, scope=None):
    return data.str.pad(length, side='left', fillchar=pad)


@execute_node.register(
    ops.RPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_rpad(op, data, length, pad, scope=None):
    return data.str.pad(length, side='right', fillchar=pad)


@execute_node.register(ops.Reverse, pd.Series)
def execute_string_reverse(op, data, scope=None):
    return data.str[::-1]


@execute_node.register(ops.Lowercase, pd.Series)
def execute_string_lower(op, data, scope=None):
    return data.str.lower()


@execute_node.register(ops.Uppercase, pd.Series)
def execute_string_upper(op, data, scope=None):
    return data.str.upper()


@execute_node.register(ops.Capitalize, pd.Series)
def execute_string_capitalize(op, data, scope=None):
    return data.str.capitalize()


@execute_node.register(ops.Repeat, pd.Series, (pd.Series,) + integer_types)
def execute_string_repeat(op, data, times, scope=None):
    return data.str.repeat(times)


@execute_node.register(
    ops.StringFind,
    pd.Series,
    (pd.Series,) + six.string_types,
    (pd.Series, type(None)) + integer_types,
    (pd.Series, type(None)) + integer_types,
)
def execute_string_contains(op, data, needle, start, end, scope=None):
    return data.str.find(needle, start, end)


@execute_node.register(
    ops.Between,
    pd.Series,
    (pd.Series, numbers.Real, str, datetime.datetime),
    (pd.Series, numbers.Real, str, datetime.datetime)
)
def execute_between(op, data, lower, upper, scope=None):
    return data.between(lower, upper)


@execute_node.register(ops.DistinctColumn, pd.Series)
def execute_series_distinct(op, data, scope=None):
    return pd.Series(data.unique(), name=data.name)


@execute_node.register(ops.Union, pd.DataFrame, pd.DataFrame)
def execute_union_dataframe_dataframe(op, left, right, scope=None):
    return pd.concat([left, right], axis=0)
