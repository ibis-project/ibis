from __future__ import annotations

import datafusion as df
import pyarrow as pa
import pyarrow.compute as pc

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.decompile import _to_snake_case
from ibis.formats.pyarrow import PyArrowType


def create_udf(op, udf, input_types, volatility="immutable", name=None):
    return df.udf(
        udf,
        input_types=list(map(PyArrowType.from_ibis, input_types)),
        return_type=PyArrowType.from_ibis(op.dtype),
        volatility=volatility,
        name=_to_snake_case(op.__name__) if name is None else name,
    )


def extract_microsecond(array: pa.Array) -> pa.Array:
    arr = pc.multiply(pc.millisecond(array), 1000)
    return pc.cast(pc.add(pc.microsecond(array), arr), pa.int32())


def epoch_seconds(array: pa.Array) -> pa.Array:
    return pc.cast(pc.divide(pc.cast(array, pa.int64()), 1000_000), pa.int32())


def extract_down(array: pa.Array) -> pa.Array:
    return pc.choose(
        pc.day_of_week(array),
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )


def extract_second(array: pa.Array) -> pa.Array:
    return pc.cast(pc.second(array), pa.int32())


def extract_millisecond(array: pa.Array) -> pa.Array:
    return pc.cast(pc.millisecond(array), pa.int32())


def extract_hour(array: pa.Array) -> pa.Array:
    return pc.cast(pc.hour(array), pa.int32())


def extract_minute(array: pa.Array) -> pa.Array:
    return pc.cast(pc.minute(array), pa.int32())


UDFS = {
    "extract_microseconds_time": create_udf(
        ops.ExtractMicrosecond,
        extract_microsecond,
        input_types=[dt.time],
        name="extract_microseconds_time",
    ),
    "extract_microsecond_timestamp": create_udf(
        ops.ExtractMicrosecond,
        extract_microsecond,
        input_types=[dt.timestamp],
        name="extract_microseconds_timestamp",
    ),
    "extract_epoch_seconds_time": create_udf(
        ops.ExtractEpochSeconds,
        epoch_seconds,
        input_types=[dt.time],
        name="extract_epoch_seconds_time",
    ),
    "extract_epoch_seconds_timestamp": create_udf(
        ops.ExtractEpochSeconds,
        epoch_seconds,
        input_types=[dt.timestamp],
        name="extract_epoch_seconds_timestamp",
    ),
    "extract_down_date": create_udf(
        ops.DayOfWeekName,
        extract_down,
        input_types=[dt.date],
        name="extract_down_date",
    ),
    "extract_down_timestamp": create_udf(
        ops.DayOfWeekName,
        extract_down,
        input_types=[dt.timestamp],
        name="extract_down_timestamp",
    ),
    "extract_second_time": create_udf(
        ops.ExtractSecond,
        extract_second,
        input_types=[dt.time],
        name="extract_second_time",
    ),
    "extract_second_timestamp": create_udf(
        ops.ExtractSecond,
        extract_second,
        input_types=[dt.timestamp],
        name="extract_second_timestamp",
    ),
    "extract_millisecond_time": create_udf(
        ops.ExtractMillisecond,
        extract_millisecond,
        input_types=[dt.time],
        name="extract_millisecond_time",
    ),
    "extract_millisecond_timestamp": create_udf(
        ops.ExtractMillisecond,
        extract_millisecond,
        input_types=[dt.timestamp],
        name="extract_millisecond_timestamp",
    ),
    "extract_hour_time": create_udf(
        ops.ExtractHour, extract_hour, input_types=[dt.time], name="extract_hour_time"
    ),
    "extract_minute_time": create_udf(
        ops.ExtractMinute,
        extract_minute,
        input_types=[dt.time],
        name="extract_minute_time",
    ),
}
