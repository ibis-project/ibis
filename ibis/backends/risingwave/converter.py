from __future__ import annotations

from ibis.formats.pandas import PandasData


class RisingWavePandasData(PandasData):
    @classmethod
    def convert_Binary(cls, s, dtype, pandas_type):
        return s.map(bytes, na_action="ignore")
