from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.backends.pyspark import ddl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd


class PySparkTable(ir.Table):
    @property
    def _qualified_name(self) -> str:
        op = self.op()
        return sg.table(
            op.name, db=op.namespace.schema, catalog=op.namespace.database, quoted=True
        ).sql(dialect="spark")

    @property
    def _database(self) -> str:
        return self.op().namespace

    @property
    def _unqualified_name(self) -> str:
        return self.name

    @property
    def name(self):
        return self.op().name

    @property
    def _client(self):
        return self.op().source

    def compute_stats(self, noscan: bool = False):
        """Invoke the Spark `ANALYZE TABLE <tbl> COMPUTE STATISTICS` command.

        Parameters
        ----------
        noscan
            If `True`, collect only basic statistics for the table (number of
            rows, size in bytes).

        See Also
        --------
        [`pyspark.Backend.compute_stats`](../backends/pyspark.qmd#ibis.backends.pyspark.Backend.compute_stats)
        """
        return self._client.compute_stats(self._qualified_name, noscan=noscan)

    def drop(self):
        """Drop the table from the database."""
        self._client.drop_table_or_view(self._qualified_name)

    def truncate(self):
        """Truncate the table, removing all data."""
        self._client.truncate_table(self._qualified_name)

    @staticmethod
    def _validate_compatible(from_schema, to_schema):
        if set(from_schema.names) != set(to_schema.names):
            raise com.IbisInputError("Schemas have different names")

        for name in from_schema:
            lt = from_schema[name]
            rt = to_schema[name]
            if not dt.castable(lt, rt):
                raise com.IbisInputError(f"Cannot safely cast {lt!r} to {rt!r}")

    def insert(
        self,
        obj: ir.Table | pd.DataFrame | None = None,
        overwrite: bool = False,
        values: Iterable[Any] | None = None,
        validate: bool = True,
    ):
        """Insert data into the table.

        Parameters
        ----------
        obj
            Table expression or pandas DataFrame
        overwrite
            If True, will replace existing contents of table
        values
            Values to insert. Not implemented currently.
        validate
            If True, do more rigorous validation that schema of table being
            inserted is compatible with the existing table

        Examples
        --------
        >>> t.insert(table_expr)  # quartodoc: +SKIP # doctest: +SKIP

        # Completely overwrite contents
        >>> t.insert(table_expr, overwrite=True)  # quartodoc: +SKIP # doctest: +SKIP
        """
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            spark_df = self._session.createDataFrame(obj)
            spark_df.insertInto(self.name, overwrite=overwrite)
            return None

        expr = obj

        if values is not None:
            raise NotImplementedError

        if validate:
            existing_schema = self.schema()
            insert_schema = expr.schema()
            if not insert_schema.equals(existing_schema):
                self._validate_compatible(insert_schema, existing_schema)

        ast = self._client.compiler.to_ast(expr)
        select = ast.queries[0]
        statement = ddl.InsertSelect(self._qualified_name, select, overwrite=overwrite)
        return self._client.raw_sql(statement.compile())

    def alter(self, tbl_properties: Mapping[str, str] | None = None) -> Any:
        """Change settings and parameters of the table.

        Parameters
        ----------
        tbl_properties
            Spark table properties
        """

        stmt = ddl.AlterTable(self._qualified_name, tbl_properties=tbl_properties)
        return self._client.raw_sql(stmt.compile())
