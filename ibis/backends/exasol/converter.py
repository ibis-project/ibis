from __future__ import annotations

import datetime

from ibis.formats.pandas import PandasData


class ExasolPandasData(PandasData):
    @classmethod
    def convert_String(cls, s, dtype, pandas_type):
        if s.dtype != "object":
            return s.map(str)
        else:
            return s

    @classmethod
    def convert_Interval(cls, s, dtype, pandas_dtype):
        def parse_timedelta(value):
            # format is '(+|-)days hour:minute:second.millisecond'
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

        if s.dtype == "int64":
            # exasol can return intervals as the number of integer days (e.g.,
            # from subtraction of two dates)
            #
            # TODO: investigate whether days are the only interval ever
            # returned as integers
            return s.map(lambda days: datetime.timedelta(days=days))
        return s.map(parse_timedelta, na_action="ignore")
