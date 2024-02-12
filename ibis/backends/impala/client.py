from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot as sg

import ibis
import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.impala import ddl
from ibis.backends.impala.ddl import AlterTable, InsertSelect

if TYPE_CHECKING:
    import pandas as pd


class ImpalaTable(ir.Table):
    """A physical table in the Impala-Hive metastore."""

    @property
    def _qualified_name(self) -> str:
        op = self.op()
        return sg.table(op.name, catalog=op.namespace.database).sql(dialect="hive")

    @property
    def _unqualified_name(self) -> str:
        return self.op().name

    @property
    def _client(self):
        return self.op().source

    @property
    def _database(self) -> str:
        return self.op().namespace.database

    def compute_stats(self, incremental=False):
        """Invoke Impala COMPUTE STATS command on the table."""
        return self._client.compute_stats(
            self.op().name, database=self._database, incremental=incremental
        )

    def invalidate_metadata(self):
        self._client.invalidate_metadata(self.op().name, database=self._database)

    def refresh(self):
        self._client.refresh(self.op().name, database=self._database)

    def metadata(self):
        """Return results of `DESCRIBE FORMATTED` statement."""
        return self._client.describe_formatted(self.op().name, database=self._database)

    describe_formatted = metadata

    def files(self):
        """Return results of SHOW FILES statement."""
        return self._client.show_files(self.op().name, database=self._database)

    def drop(self):
        """Drop the table from the database."""
        self._client.drop_table_or_view(self.op().name, database=self._database)

    def truncate(self):
        self._client.truncate_table(self.op().name, database=self._database)

    def insert(
        self,
        obj=None,
        overwrite=False,
        partition=None,
        values=None,
        validate=True,
    ):
        """Insert into an Impala table.

        Parameters
        ----------
        obj
            Table expression or DataFrame
        overwrite
            If True, will replace existing contents of table
        partition
            For partitioned tables, indicate the partition that's being
            inserted into, either with an ordered list of partition keys or a
            dict of partition field name to value. For example for the
            partition (year=2007, month=7), this can be either (2007, 7) or
            {'year': 2007, 'month': 7}.
        values
            Unsupported and unused
        validate
            If True, do more rigorous validation that schema of table being
            inserted is compatible with the existing table

        Examples
        --------
        Append to an existing table

        >>> t.insert(table_expr)  # quartodoc: +SKIP # doctest: +SKIP

        Completely overwrite contents

        >>> t.insert(table_expr, overwrite=True)  # quartodoc: +SKIP # doctest: +SKIP

        """
        if values is not None:
            raise NotImplementedError

        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        self._client._run_pre_execute_hooks(obj)

        expr = obj
        if validate:
            existing_schema = self.schema()
            insert_schema = expr.schema()
            if not insert_schema.equals(existing_schema):
                _validate_compatible(insert_schema, existing_schema)

        if partition is not None:
            partition_schema = self.partition_schema()
            partition_schema_names = frozenset(partition_schema.names)
            expr = expr.select(
                [
                    column
                    for column in expr.columns
                    if column not in partition_schema_names
                ]
            )
        else:
            partition_schema = None

        statement = InsertSelect(
            self._qualified_name,
            self._client.compile(expr),
            partition=partition,
            partition_schema=partition_schema,
            overwrite=overwrite,
        )
        self._client._safe_exec_sql(statement.compile())
        return self

    def load_data(self, path, overwrite=False, partition=None):
        """Load data into an Impala table.

        Parameters
        ----------
        path
            Data to load
        overwrite
            Overwrite the existing data in the entire table or indicated
            partition
        partition
            If specified, the partition must already exist

        """
        if partition is not None:
            partition_schema = self.partition_schema()
        else:
            partition_schema = None

        stmt = ddl.LoadData(
            self._qualified_name,
            path,
            partition=partition,
            partition_schema=partition_schema,
            overwrite=overwrite,
        )

        self._client._safe_exec_sql(stmt.compile())
        return self

    @property
    def name(self) -> str:
        return self.op().name

    @property
    def is_partitioned(self):
        """True if the table is partitioned."""
        return self.metadata().is_partitioned

    def partition_schema(self):
        """Return the schema for the partition columns."""
        schema = self.schema()
        result = self.partitions()

        partition_fields = []
        for col in result.columns:
            if col not in schema:
                break
            partition_fields.append((col, schema[col]))

        return sch.Schema(dict(partition_fields))

    def add_partition(self, spec, location=None):
        """Add a new table partition.

        Partition parameters can be set in a single DDL statement or you can
        use `alter_partition` to set them after the fact.
        """
        part_schema = self.partition_schema()
        stmt = ddl.AddPartition(
            self._qualified_name, spec, part_schema, location=location
        )
        self._client._safe_exec_sql(stmt)
        return self

    def alter(
        self,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        """Change settings and parameters of the table.

        Parameters
        ----------
        location
            For partitioned tables, you may want the alter_partition function
        format
            Table format
        tbl_properties
            Table properties
        serde_properties
            Serialization/deserialization properties

        """

        def _run_ddl(**kwds):
            stmt = AlterTable(self._qualified_name, **kwds)
            self._client._safe_exec_sql(stmt)
            return self

        return self._alter_table_helper(
            _run_ddl,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )

    def set_external(self, is_external=True):
        """Toggle the `EXTERNAL` table property."""
        self.alter(tbl_properties={"EXTERNAL": is_external})

    def alter_partition(
        self,
        spec,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        """Change settings and parameters of an existing partition.

        Parameters
        ----------
        spec
            The partition keys for the partition being modified
        location
            Location of the partition
        format
            Table format
        tbl_properties
            Table properties
        serde_properties
            Serialization/deserialization properties

        """
        part_schema = self.partition_schema()

        def _run_ddl(**kwds):
            stmt = ddl.AlterPartition(self._qualified_name, spec, part_schema, **kwds)
            self._client._safe_exec_sql(stmt)
            return self

        return self._alter_table_helper(
            _run_ddl,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )

    def _alter_table_helper(self, f, **alterations):
        results = []
        for k, v in alterations.items():
            if v is None:
                continue
            result = f(**{k: v})
            results.append(result)
        return results

    def drop_partition(self, spec):
        """Drop an existing table partition."""
        part_schema = self.partition_schema()
        stmt = ddl.DropPartition(self._qualified_name, spec, part_schema)
        self._client._safe_exec_sql(stmt)
        return self

    def partitions(self):
        """Return information about the table's partitions.

        Raises an exception if the table is not partitioned.
        """
        return self._client.list_partitions(self._qualified_name)

    def stats(self) -> pd.DataFrame:
        """Return results of `SHOW TABLE STATS`.

        If not partitioned, contains only one row.

        Returns
        -------
        DataFrame
            Table statistics

        """
        return self._client.table_stats(self._qualified_name)

    def column_stats(self) -> pd.DataFrame:
        """Return results of `SHOW COLUMN STATS`.

        Returns
        -------
        DataFrame
            Column statistics

        """
        return self._client.column_stats(self._qualified_name)


# ----------------------------------------------------------------------
# ORM-ish usability layer


class ScalarFunction:
    def drop(self):
        pass


class AggregateFunction:
    def drop(self):
        pass


def _validate_compatible(from_schema, to_schema):
    if set(from_schema.names) != set(to_schema.names):
        raise com.IbisInputError("Schemas have different names")

    for name in from_schema:
        lt = from_schema[name]
        rt = to_schema[name]
        if not lt.castable(rt):
            raise com.IbisInputError(f"Cannot safely cast {lt!r} to {rt!r}")
