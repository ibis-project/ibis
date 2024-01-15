from __future__ import annotations

import datetime

from ibis.common.temporal import normalize_timezone
from ibis.formats.pandas import PandasData


class PySparkPandasData(PandasData):
    @classmethod
    def convert_Time(cls, s, dtype, pandas_type):
        def convert(timedelta):
            comps = timedelta.components
            return datetime.time(
                hour=comps.hours,
                minute=comps.minutes,
                second=comps.seconds,
                microsecond=comps.milliseconds * 1000 + comps.microseconds,
            )

        return s.map(convert, na_action="ignore")

    @classmethod
    def convert_Timestamp_element(cls, dtype):
        if dtype.timezone is None:
            tz = normalize_timezone("UTC")

            def converter(value):
                try:
                    return value.astimezone(tz).replace(tzinfo=None)
                except TypeError:
                    return value.tz_localize(tz).replace(tzinfo=None)
        else:
            tz = normalize_timezone(dtype.timezone)

            def converter(value):
                try:
                    return value.astimezone(tz)
                except TypeError:
                    return value.tz_localize(tz)

        return converter
