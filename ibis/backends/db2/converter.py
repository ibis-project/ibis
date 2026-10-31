from __future__ import annotations

from ibis.formats.pandas import PandasData


class Db2PandasData(PandasData):
    @classmethod
    def convert_Boolean(cls, s, dtype, pandas_type):
        # ibm_db_dbi returns BOOLEAN as int (0/1) in an object-dtype series,
        # which the base class returns unchanged.
        if s.empty:
            return s.astype(pandas_type)
        return s.map(bool, na_action="ignore")
