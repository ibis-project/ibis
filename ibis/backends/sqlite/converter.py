from __future__ import annotations

from packaging.version import parse as vparse

from ibis.formats.pandas import PandasData


class SQLitePandasData(PandasData):
    @classmethod
    def convert_Timestamp(cls, s, dtype, pandas_type):
        """A more flexible timestamp parser.

        This handles the valid formats supported by SQLite.
        See https://sqlite.org/lang_datefunc.html#time_values for more info.
        """
        try:
            return super().convert_Timestamp(s, dtype, pandas_type)
        except ValueError:
            import pandas as pd

            # Parsing failed, try a more relaxed parser
            format = "mixed" if vparse(pd.__version__) >= vparse("2.0.0") else None
            return pd.to_datetime(s, format=format, utc=True)
