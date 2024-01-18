from __future__ import annotations

from ibis.formats.pandas import PandasData


class ExasolPandasData(PandasData):
    @classmethod
    def convert_String(cls, s, dtype, pandas_type):
        if s.dtype != "object":
            return s.map(str)
        else:
            return s
