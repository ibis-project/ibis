from __future__ import annotations

import pandas as pd

from ibis.formats.pandas import PandasData


class OraclePandasData(PandasData):
    @classmethod
    def convert_Timestamp_element(cls, dtype):
        return pd.Timestamp.fromisoformat

    @classmethod
    def convert_Date_element(cls, dtype):
        return pd.Timestamp.fromisoformat

    @classmethod
    def convert_Time_element(cls, dtype):
        return pd.Timestamp.fromisoformat
