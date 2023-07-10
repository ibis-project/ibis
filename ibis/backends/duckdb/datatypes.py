from __future__ import annotations

import numpy as np

from ibis.formats.pandas import PandasData


class DuckDBPandasData(PandasData):
    @staticmethod
    def convert_Array(s, dtype, pandas_type):
        return s.replace(np.nan, None)
