from __future__ import annotations

import datetime

import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
from pandas import Timedelta

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    TypeRegistrationDict,
    make_selected_obj,
    register_types_to_dispatcher,
)
from ibis.backends.pandas.core import (
    date_types,
    integer_types,
    numeric_types,
    timedelta_types,
    timestamp_types,
)
from ibis.backends.pandas.execution.temporal import (
    day_name,
    execute_cast_integer_to_interval_series,
    execute_date_add,
    execute_date_sub_diff,
    execute_date_sub_diff_date_series,
    execute_date_sub_diff_series_date,
    execute_day_of_week_index_series,
    execute_day_of_week_name_series,
    execute_epoch_seconds,
    execute_extract_microsecond_series,
    execute_extract_millisecond_series,
    execute_extract_timestamp_field_series,
    execute_interval_add_multiply_delta_series,
    execute_interval_from_integer_series,
    execute_interval_multiply_fdiv_series_numeric,
    execute_strftime_series_str,
    execute_timestamp_add_datetime_series,
    execute_timestamp_date,
    execute_timestamp_diff_series_datetime,
    execute_timestamp_diff_sub_datetime_series,
    execute_timestamp_diff_sub_series_series,
    execute_timestamp_from_unix,
    execute_timestamp_interval_add_series_delta,
    execute_timestamp_interval_add_series_series,
    execute_timestamp_sub_series_timedelta,
    execute_timestamp_truncate,
)

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.Cast: [((dd.Series, dt.Interval), execute_cast_integer_to_interval_series)],
    ops.Strftime: [((dd.Series, str), execute_strftime_series_str)],
    ops.TimestampFromUNIX: [
        (((dd.Series,) + integer_types), execute_timestamp_from_unix)
    ],
    ops.ExtractTemporalField: [((dd.Series,), execute_extract_timestamp_field_series)],
    ops.ExtractMicrosecond: [((dd.Series,), execute_extract_microsecond_series)],
    ops.ExtractMillisecond: [((dd.Series,), execute_extract_millisecond_series)],
    ops.ExtractEpochSeconds: [((dd.Series,), execute_epoch_seconds)],
    ops.IntervalFromInteger: [((dd.Series,), execute_interval_from_integer_series)],
    ops.IntervalAdd: [
        (
            (timedelta_types, dd.Series),
            execute_interval_add_multiply_delta_series,
        ),
        (
            (dd.Series, timedelta_types),
            execute_timestamp_interval_add_series_delta,
        ),
        ((dd.Series, dd.Series), execute_timestamp_interval_add_series_series),
    ],
    ops.IntervalSubtract: [
        ((dd.Series, dd.Series), execute_timestamp_diff_sub_series_series)
    ],
    ops.IntervalMultiply: [
        (
            (timedelta_types, numeric_types + (dd.Series,)),
            execute_interval_add_multiply_delta_series,
        ),
        (
            (dd.Series, numeric_types + (dd.Series,)),
            execute_interval_multiply_fdiv_series_numeric,
        ),
    ],
    ops.IntervalFloorDivide: [
        (
            (
                (Timedelta, dd.Series),
                numeric_types + (dd.Series,),
            ),
            execute_interval_multiply_fdiv_series_numeric,
        )
    ],
    ops.TimestampAdd: [
        ((timestamp_types, dd.Series), execute_timestamp_add_datetime_series),
        (
            (dd.Series, timedelta_types),
            execute_timestamp_interval_add_series_delta,
        ),
        ((dd.Series, dd.Series), execute_timestamp_interval_add_series_series),
    ],
    ops.TimestampSub: [
        ((dd.Series, timedelta_types), execute_timestamp_sub_series_timedelta),
        (
            (timestamp_types, dd.Series),
            execute_timestamp_diff_sub_datetime_series,
        ),
    ],
    (ops.TimestampDiff, ops.TimestampSub): [
        ((dd.Series, dd.Series), execute_timestamp_diff_sub_series_series)
    ],
    ops.TimestampDiff: [
        ((dd.Series, timestamp_types), execute_timestamp_diff_series_datetime),
        (
            (timestamp_types, dd.Series),
            execute_timestamp_diff_sub_datetime_series,
        ),
    ],
    ops.DayOfWeekIndex: [((dd.Series,), execute_day_of_week_index_series)],
    ops.DayOfWeekName: [((dd.Series,), execute_day_of_week_name_series)],
    ops.Date: [((dd.Series,), execute_timestamp_date)],
    ops.DateAdd: [
        ((dd.Series, timedelta_types), execute_date_add),
        ((timedelta_types, dd.Series), execute_date_add),
        ((dd.Series, dd.Series), execute_date_add),
        ((date_types, dd.Series), execute_date_add),
        ((dd.Series, date_types), execute_date_add),
    ],
    ops.DateSub: [
        ((date_types, dd.Series), execute_date_sub_diff),
        ((dd.Series, dd.Series), execute_date_sub_diff),
        ((dd.Series, timedelta_types), execute_date_sub_diff),
    ],
    ops.DateDiff: [
        ((date_types, dd.Series), execute_date_sub_diff_date_series),
        ((dd.Series, dd.Series), execute_date_sub_diff),
        ((dd.Series, date_types), execute_date_sub_diff_series_date),
    ],
    ops.TimestampTruncate: [((dd.Series,), execute_timestamp_truncate)],
    ops.DateTruncate: [((dd.Series,), execute_timestamp_truncate)],
}
register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


@execute_node.register(
    ops.BetweenTime,
    dd.Series,
    (dd.Series, str, datetime.time),
    (dd.Series, str, datetime.time),
)
def execute_between_time(op, data, lower, upper, **kwargs):
    if getattr(data.dtype, "tz", None) is not None:
        localized = data.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        localized = data

    time = localized.dt.time.astype(str)
    indexer = ((time >= lower) & (time <= upper)).to_dask_array(True)

    result = da.zeros(len(data), dtype=np.bool_)
    result[indexer] = True
    return dd.from_array(result)


@execute_node.register(ops.DayOfWeekIndex, ddgb.SeriesGroupBy)
def execute_day_of_week_index_series_group_by(op, data, **kwargs):
    return make_selected_obj(data).dt.dayofweek.astype(np.int16).groupby(data.index)


@execute_node.register(ops.DayOfWeekName, ddgb.SeriesGroupBy)
def execute_day_of_week_name_series_group_by(op, data, **kwargs):
    return day_name(make_selected_obj(data).dt).groupby(data.index)
