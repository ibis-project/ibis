from __future__ import annotations

import pandas as pd

from ibis.formats.pandas import PandasData

# The "mixed" format was added in pandas 2
_DATETIME_FORMAT = "mixed" if pd.__version__ >= "2.0.0" else None


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
            # Parsing failed, try a more relaxed parser
            return pd.to_datetime(s, format=_DATETIME_FORMAT, utc=True)
