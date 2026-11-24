from __future__ import annotations

import datetime

import pandas as pd
import pandas.api.types as pdt

from ibis.formats.pandas import PandasData


def _holds_objects_or_strings(s: pd.Series) -> bool:
    # a column of strings and NULLs has the object dtype on pandas < 3 and a
    # string dtype on pandas >= 3
    return pdt.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.StringDtype)


class ExasolPandasData(PandasData):
    @classmethod
    def convert_String(cls, s, dtype, pandas_type):
        if not _holds_objects_or_strings(s):
            return s.map(str)
        else:
            return s

    @classmethod
    def convert_Int64(cls, s, dtype, pandas_dtype):
        if _holds_objects_or_strings(s):
            # exasol returns BIGINT types as strings (or None for NULL).
            # s.astype("int64") will fail in this case, using `Series.map`
            # is the best we can do.
            return s.map(int, na_action="ignore")
        return s if s.dtype == pandas_dtype else s.astype(pandas_dtype)

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
