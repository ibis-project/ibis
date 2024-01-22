from __future__ import annotations

import datetime

from ibis.formats.pandas import PandasData


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
