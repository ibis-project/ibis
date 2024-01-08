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
        def converter(value, dtype=dtype):
            if (tz := dtype.timezone) is not None:
                return value.astimezone(normalize_timezone(tz))

            return value.astimezone(normalize_timezone("UTC")).replace(tzinfo=None)

        return converter
