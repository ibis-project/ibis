import six
import datetime
import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.pandas.dispatch import execute_node
from ibis.pandas.core import numeric_types


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
    return getattr(data.dt, field_name).astype(np.int32)


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


@execute_node.register(ops.Date, pd.Series)
def execute_timestamp_date(op, data, **kwargs):
    return data.dt.floor('d')


@execute_node.register((ops.TimestampTruncate, ops.DateTruncate), pd.Series)
def execute_timestamp_truncate(op, data, **kwargs):
    dtype = 'datetime64[{}]'.format(op.unit)
    array = data.values.astype(dtype)
    return pd.Series(array, name=data.name)


@execute_node.register(ops.IntervalFromInteger, pd.Series)
def execute_interval_from_integer_series(op, data, **kwargs):
    resolution = '{}s'.format(op.resolution)

    def convert_to_offset(n):
        return pd.offsets.DateOffset(**{resolution: n})

    return data.apply(convert_to_offset)


@execute_node.register(
    ops.TimestampAdd,
    (datetime.datetime, pd.Series),
    (datetime.timedelta, np.timedelta64, pd.Timedelta, pd.Series)
)
@execute_node.register(
    ops.IntervalAdd, (pd.Timedelta, pd.Series), (pd.Timedelta, pd.Series)
)
def execute_timestamp_add_delta(op, left, right, **kwargs):
    return left + right


@execute_node.register(
    (ops.TimestampSub, ops.TimestampDiff),
    (datetime.datetime, pd.Series),
    (
        datetime.datetime,
        np.datetime64,
        datetime.timedelta,
        np.timedelta64,
        pd.Timedelta,
        pd.Series
    ),
)
def execute_timestamp_sub_diff(op, left, right, **kwargs):
    return left - right


@execute_node.register(
    ops.IntervalMultiply,
    (pd.Timedelta, pd.Series), numeric_types + (pd.Series,)
)
def execute_interval_multiply(op, left, right, **kwargs):
    return left * right


@execute_node.register(
    ops.IntervalFloorDivide,
    (pd.Timedelta, pd.Series),
    numeric_types + (pd.Series,)
)
def execute_interval_floor_divide(op, left, right, **kwargs):
    return left // right
