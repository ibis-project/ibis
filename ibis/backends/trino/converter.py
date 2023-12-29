from __future__ import annotations

import datetime

from ibis.formats.pandas import PandasData


class TrinoPandasData(PandasData):
    @classmethod
    def convert_Interval(cls, s, dtype, pandas_dtype):
        def parse_trino_timedelta(value):
            # format is 'days hour:minute:second.millisecond'
            days, rest = value.split(" ", 1)
            hms, millis = rest.split(".", 1)
            hours, minutes, seconds = hms.split(":")
            return datetime.timedelta(
                days=int(days),
                hours=int(hours),
                minutes=int(minutes),
                seconds=int(seconds),
                milliseconds=int(millis),
            )

        return s.map(parse_trino_timedelta, na_action="ignore")
