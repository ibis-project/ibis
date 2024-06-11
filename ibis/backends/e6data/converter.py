from __future__ import annotations

import datetime

from ibis.formats.pandas import PandasData


class MySQLPandasData(PandasData):
    # TODO(kszucs): this could be reused at other backends, like pyspark
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
    def convert_Timestamp(cls, s, dtype, pandas_type):
        if s.dtype == "object":
            s = s.replace("0000-00-00 00:00:00", None)
        return super().convert_Timestamp(s, dtype, pandas_type)
