from __future__ import annotations

import numpy as np
from ibis.formats.pandas import PandasData


class DuckDBPandasData(PandasData):
    @staticmethod
    def convert_Map(s, dtype, pandas_type):
        return s.map(lambda x: dict(zip(x["key"], x["value"])), na_action="ignore")

    @staticmethod
    def convert_Array(s, dtype, pandas_type):
        return s.replace(np.nan, None)
