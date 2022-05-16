import re
from typing import Any

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.types as ir

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


class ClickhouseTable(ir.Table):
    """References a physical table in Clickhouse"""

    @property
    def _qualified_name(self):
        return self.op().name

    @property
    def _client(self):
        return self.op().source

    def invalidate_metadata(self):
        self._client.invalidate_metadata(self._qualified_name)

    def metadata(self) -> Any:
        """Return the parsed results of a `DESCRIBE FORMATTED` statement.

        Returns
        -------
        TableMetadata
            Table metadata
        """
        return self._client.describe_formatted(self._qualified_name)

    describe_formatted = metadata

    @property
    def name(self):
        return self.op().name

    def insert(self, obj, **kwargs):
        from ibis.backends.clickhouse.identifiers import quote_identifier

        schema = self.schema()

        assert isinstance(obj, pd.DataFrame)
        assert set(schema.names) >= set(obj.columns)

        columns = ", ".join(map(quote_identifier, obj.columns))
        query = f"INSERT INTO {self._qualified_name} ({columns}) VALUES"

        # convert data columns with datetime64 pandas dtype to native date
        # because clickhouse-driver 0.0.10 does arithmetic operations on it
        obj = obj.copy()
        for col in obj.select_dtypes(include=[np.datetime64]):
            if isinstance(schema[col], dt.Date):
                obj[col] = obj[col].dt.date

        settings = kwargs.pop("settings", {})
        settings["use_numpy"] = True
        return self._client.con.insert_dataframe(
            query,
            obj,
            settings=settings,
            **kwargs,
        )
