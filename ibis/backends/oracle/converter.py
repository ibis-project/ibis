from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    from ibis.expr.schema import Schema


class OraclePandasData(PandasData):
    @classmethod
    def convert_Timestamp_element(cls, dtype):
        return datetime.datetime.fromisoformat

    @classmethod
    def convert_Date_element(cls, dtype):
        return datetime.date.fromisoformat

    @classmethod
    def convert_Time_element(cls, dtype):
        return datetime.time.fromisoformat
