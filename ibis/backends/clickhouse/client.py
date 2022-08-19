import re
from typing import Any

import pandas as pd

import ibis.expr.types as ir
from ibis import util

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

        tmpname = f"tmp_{util.guid()}"
        name = self._qualified_name
        query = f"INSERT INTO {name} SELECT * FROM {tmpname}"
        return self._client.raw_sql(query, external_tables={tmpname: obj})
