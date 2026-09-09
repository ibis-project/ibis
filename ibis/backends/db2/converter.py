"""Db2-specific pandas data converter."""

from __future__ import annotations

import pandas as pd

from ibis.formats.pandas import PandasData


class Db2PandasData(PandasData):
    """Convert raw ``ibm_db_dbi`` cursor rows to a typed pandas DataFrame.

    ``ibm_db_dbi`` returns:

    - Integers    → Python ``int``          (base class handles)
    - Floats      → Python ``float``        (base class handles)
    - Decimals    → ``decimal.Decimal``     (base class handles)
    - Strings     → Python ``str``          (base class handles)
    - Booleans    → Python ``int`` (0/1)    (override required — see convert_Boolean)
    - DATE        → ``datetime.date``       (base class handles)
    - TIME        → ``datetime.time``       (base class handles)
    - TIMESTAMP   → ``datetime.datetime``   (override required — see convert_Timestamp_element)
    """

    @classmethod
    def convert_Boolean(cls, s, dtype, pandas_type):
        """Convert a BOOLEAN column from ibm_db_dbi.

        ``ibm_db_dbi`` surfaces Db2 BOOLEAN values as Python ``int`` (0/1),
        which pandas loads as an ``object``-dtype series.  The base class
        short-circuits on ``object`` dtype and returns the series unchanged,
        leaving ints instead of bools.  This override forces the ``map(bool)``
        conversion regardless of dtype.
        """
        if s.empty:
            return s.astype(pandas_type)
        return s.map(bool, na_action="ignore")

    @classmethod
    def convert_Timestamp_element(cls, dtype):
        """Return a converter for a single timestamp value from ibm_db_dbi.

        ``ibm_db_dbi`` returns ``datetime.datetime`` objects; this converts
        them to ``pd.Timestamp`` which is what ibis expects.
        """
        return pd.Timestamp.fromisoformat
