from __future__ import annotations

import contextlib
import datetime

from ibis.common.temporal import normalize_timezone
from ibis.formats.pandas import PandasData


class ClickHousePandasData(PandasData):
    @classmethod
    def convert_Timestamp_element(cls, dtype):
        def converter(value, dtype=dtype):
            if value is None:
                return value

            with contextlib.suppress(AttributeError):
                value = value.item()

            if isinstance(value, int):
                # this can only mean a numpy or pandas timestamp because they
                # both support nanosecond precision
                #
                # when the precision is less than or equal to the value
                # supported by Python datetime.dateimte a call to .item() will
                # return a datetime.datetime but when the precision is higher
                # than the value supported by Python the value is an integer
                #
                # TODO: can we do better than implicit truncation to microseconds?
                import dateutil

                value = datetime.datetime.fromtimestamp(value / 1e9, dateutil.tz.UTC)

            if (tz := dtype.timezone) is not None:
                ntz = normalize_timezone(tz)

                # deal with this madness
                # https://github.com/ClickHouse/clickhouse-connect/pull/311
                if tz == "UTC" and value.tzinfo is None:
                    return value.replace(tzinfo=ntz)
                else:
                    return value.astimezone(ntz)

            return value.replace(tzinfo=None)

        return converter
