"""RisingWave backend."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

import psycopg2
import sqlglot as sg
import sqlglot.expressions as sge
from psycopg2 import extras

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.postgres import Backend as PostgresBackend
from ibis.backends.sql.compilers import RisingWaveCompiler
from ibis.util import experimental

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa


def data_and_encode_format(data_format, encode_format, encode_properties):
    res = ""
    if data_format is not None:
        res = res + " FORMAT " + data_format.upper()
    if encode_format is not None:
        res = res + " ENCODE " + encode_format.upper()
        if encode_properties is not None:
            res = res + " " + format_properties(encode_properties)
    return res


def format_properties(props):
    tokens = []
    for k, v in props.items():
        tokens.append(f"{k}='{v}'")
    return "( {} ) ".format(", ".join(tokens))


class Backend(PostgresBackend):
    name = "risingwave"
    compiler = RisingWaveCompiler()
    supports_python_udfs = False

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create an Ibis client connected to RisingWave database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number
        database
            Database to connect to
        schema
            RisingWave schema to use. If `None`, use the default `search_path`.

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_RISINGWAVE_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_RISINGWAVE_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_RISINGWAVE_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_RISINGWAVE_DATABASE", "dev")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        RisingWaveTable[table]
          name: functional_alltypes
          schema:
            id : int32
            bool_col : boolean
            tinyint_col : int16
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32

        """

        self.con = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            options=(f"-csearch_path={schema}" * (schema is not None)) or None,
        )

        with self.begin() as cur:
            cur.execute("SET TIMEZONE = UTC")
            cur.execute("SET RW_IMPLICIT_FLUSH TO true;")

    def create_table(
        self,
        name: str,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        # TODO(Kexiang): add `append only`
        connector_properties: dict | None = None,
        data_format: str | None = None,
        encode_format: str | None = None,
        encode_properties: dict | None = None,
    ):
        """Create a table in RisingWave.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        connector_properties
            The properties of the sink connector, providing the connector settings to push to the downstream data sink.
            Refer https://docs.risingwave.com/docs/current/data-delivery/ for the required properties of different data sink.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.

        Returns
        -------
        Table
            Table expression
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if connector_properties is not None and (
            encode_format is None or data_format is None
        ):
            raise com.UnsupportedOperationError(
                "When creating tables with connector, both encode_format and data_format are required"
            )

        properties = []

        if temp:
            raise com.UnsupportedOperationError(
                f"Creating temp tables is not supported by {self.name}"
            )

        temp_memtable_view = None
        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
                temp_memtable_view = table.op().name
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self._to_sqlglot(table)
        else:
            query = None

        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=self.compiler.quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for colname, typ in (schema or table.schema()).items()
        ]

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, db=database, quoted=self.compiler.quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        if connector_properties is None:
            create_stmt = sge.Create(
                kind="TABLE",
                this=target,
                properties=sge.Properties(expressions=properties),
            )
        else:
            create_stmt = sge.Create(
                kind="TABLE",
                this=target,
                properties=sge.Properties(
                    expressions=sge.Properties.from_dict(connector_properties)
                ),
            )
            create_stmt = create_stmt.sql(self.dialect) + data_and_encode_format(
                data_format, encode_format, encode_properties
            )

        this = sg.table(name, db=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table, expression=query).sql(self.dialect)
                cur.execute(insert_stmt)

            if overwrite:
                self.drop_table(name, database=database, force=True)
                cur.execute(
                    f"ALTER TABLE {table.sql(self.dialect)} RENAME TO {this.sql(self.dialect)}"
                )

        if schema is None:
            # Clean up temporary memtable if we've created one
            # for in-memory reads
            if temp_memtable_view is not None:
                self.drop_table(temp_memtable_view)
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [
                            sg.exp.ColumnConstraint(
                                kind=sg.exp.NotNullColumnConstraint()
                            )
                        ]
                    ),
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
            )
            create_stmt_sql = create_stmt.sql(self.dialect)

            df = op.data.to_frame()
            data = df.itertuples(index=False)
            sql = self._build_insert_template(
                name, schema=schema, columns=True, placeholder="%s"
            )
            with self.begin() as cur:
                cur.execute(create_stmt_sql)
                extras.execute_batch(cur, sql, data, 128)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        dbs = "SHOW SCHEMAS"

        with self._safe_raw_sql(dbs) as cur:
            databases = list(map(itemgetter(0), cur))

        return self._filter_with_like(databases, like)

    @experimental
    def create_materialized_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a materialized view. Materialized views can be accessed like a normal table.

        Parameters
        ----------
        name
            Materialized view name to Create.
        obj
            The select statement to materialize.
        database
            Name of the database where the view exists, if not the default
        overwrite
            Whether to overwrite the existing materialized view with the same name

        Returns
        -------
        Table
            Table expression
        """
        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, db=database, quoted=self.compiler.quoted)

        create_stmt = sge.Create(
            this=table,
            kind="MATERIALIZED VIEW",
            expression=self.compile(obj),
        )
        self._register_in_memory_tables(obj)

        with self._safe_raw_sql(create_stmt) as cur:
            if overwrite:
                target = sg.table(name, db=database).sql(self.dialect)

                self.drop_materialized_view(target, database=database, force=True)

                cur.execute(
                    f"ALTER MATERIALIZED VIEW {table.sql(self.dialect)} RENAME TO {target}"
                )

        return self.table(name, database=database)

    def drop_materialized_view(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a materialized view.

        Parameters
        ----------
        name
            Materialized view name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="MATERIALIZED VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def create_source(
        self,
        name: str,
        schema: ibis.Schema,
        *,
        database: str | None = None,
        connector_properties: dict,
        data_format: str,
        encode_format: str,
        encode_properties: dict | None = None,
    ) -> ir.Table:
        """Creating a source.

        Parameters
        ----------
        name
            Source name to Create.
        schema
            The schema for the new Source.
        database
            Name of the database where the source exists, if not the default.
        connector_properties
            The properties of the source connector, providing the connector settings to access the upstream data source.
            Refer https://docs.risingwave.com/docs/current/data-ingestion/ for the required properties of different data source.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.

        Returns
        -------
        Table
            Table expression
        """
        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=self.compiler.quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for colname, typ in schema.items()
        ]

        table = sg.table(name, db=database, quoted=self.compiler.quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="SOURCE",
            this=target,
            properties=sge.Properties(
                expressions=sge.Properties.from_dict(connector_properties)
            ),
        )

        create_stmt = create_stmt.sql(self.dialect) + data_and_encode_format(
            data_format, encode_format, encode_properties
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, database=database)

    def drop_source(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a Source.

        Parameters
        ----------
        name
            Source name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the source does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="SOURCE",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def create_sink(
        self,
        name: str,
        sink_from: str | None = None,
        connector_properties: dict | None = None,
        *,
        obj: ir.Table | None = None,
        database: str | None = None,
        data_format: str | None = None,
        encode_format: str | None = None,
        encode_properties: dict | None = None,
    ) -> None:
        """Creating a sink.

        Parameters
        ----------
        name
            Sink name to Create.
        sink_from
            The table or materialized view name to sink from. Only one of `sink_from` or `obj` can be
            provided.
        connector_properties
            The properties of the sink connector, providing the connector settings to push to the downstream data sink.
            Refer https://docs.risingwave.com/docs/current/data-delivery/ for the required properties of different data sink.
        obj
            An Ibis table expression that will be used to extract the schema and the data of the new table. Only one of `sink_from` or `obj` can be provided.
        database
            Name of the database where the source exists, if not the default.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.
        """
        table = sg.table(name, db=database, quoted=self.compiler.quoted)
        if sink_from is None and obj is None:
            raise ValueError("Either `sink_from` or `obj` must be specified")
        if sink_from is not None and obj is not None:
            raise ValueError("Only one of `sink_from` or `obj` can be specified")

        if (encode_format is None) != (data_format is None):
            raise com.UnsupportedArgumentError(
                "When creating sinks, both encode_format and data_format must be provided, or neither should be"
            )

        if sink_from is not None:
            create_stmt = f"CREATE SINK {table.sql(self.dialect)} FROM {sink_from}"
        else:
            create_stmt = sge.Create(
                this=table,
                kind="SINK",
                expression=self.compile(obj),
            ).sql(self.dialect)
        create_stmt = (
            create_stmt
            + " WITH "
            + format_properties(connector_properties)
            + data_and_encode_format(data_format, encode_format, encode_properties)
        )
        with self._safe_raw_sql(create_stmt):
            pass

    def drop_sink(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a Sink.

        Parameters
        ----------
        name
            Sink name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the source does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="SINK",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass
