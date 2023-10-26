from __future__ import annotations

import itertools
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import pyarrow.compute as pc

import ibis.expr.datatypes as dt  # noqa: TCH001


def _extract_epoch_seconds(array) -> dt.int32:
    return pc.cast(pc.divide(pc.cast(array, pa.int64()), 1_000_000), pa.int32())


def extract_epoch_seconds_date(array: dt.date) -> dt.int32:
    return _extract_epoch_seconds(array)


def extract_epoch_seconds_timestamp(array: dt.Timestamp(scale=6)) -> dt.int32:
    return _extract_epoch_seconds(array)


def _extract_second(array):
    return pc.cast(pc.second(array), pa.int32())


def extract_second_timestamp(array: dt.Timestamp(scale=9)) -> dt.int32:
    return _extract_second(array)


def extract_second_time(array: dt.time) -> dt.int32:
    return _extract_second(array)


def _extract_millisecond(array) -> dt.int32:
    return pc.cast(pc.millisecond(array), pa.int32())


def extract_millisecond_timestamp(array: dt.Timestamp(scale=9)) -> dt.int32:
    return _extract_millisecond(array)


def extract_millisecond_time(array: dt.time) -> dt.int32:
    return _extract_millisecond(array)


def extract_microsecond(array: dt.Timestamp(scale=9)) -> dt.int32:
    arr = pc.multiply(pc.millisecond(array), 1000)
    return pc.cast(pc.add(pc.microsecond(array), arr), pa.int32())


def _extract_query_arrow(
    arr: pa.StringArray, *, param: str | None = None
) -> pa.StringArray:
    if param is None:

        def _extract_query(url, param):
            return urlsplit(url).query

        params = itertools.repeat(None)
    else:

        def _extract_query(url, param):
            query = urlsplit(url).query
            value = parse_qs(query)[param]
            return value if len(value) > 1 else value[0]

        params = param.to_pylist()

    return pa.array(map(_extract_query, arr.to_pylist(), params))


def extract_query(array: str) -> str:
    return _extract_query_arrow(array)


def extract_query_param(array: str, param: str) -> str:
    return _extract_query_arrow(array, param=param)


def extract_user_info(arr: str) -> str:
    def _extract_user_info(url):
        url_parts = urlsplit(url)
        username = url_parts.username or ""
        password = url_parts.password or ""
        return f"{username}:{password}"

    return pa.array(map(_extract_user_info, arr.to_pylist()))


def extract_url_field(arr: str, field: str) -> str:
    field = field.to_pylist()[0]
    return pa.array(getattr(url, field, "") for url in map(urlsplit, arr.to_pylist()))


def sign(arr: dt.float64) -> dt.float64:
    return pc.sign(arr)


def _extract_minute(array) -> dt.int32:
    return pc.cast(pc.minute(array), pa.int32())


def extract_minute_time(array: dt.time) -> dt.int32:
    return _extract_minute(array)


def extract_minute_timestamp(array: dt.Timestamp(scale=9)) -> dt.int32:
    return _extract_minute(array)


def extract_hour_time(array: dt.time) -> dt.int32:
    return pc.cast(pc.hour(array), pa.int32())
