"""Materialize backend."""

from __future__ import annotations

from typing import Any

import sqlglot as sg
from sqlglot import expressions as sge

import ibis
import ibis.expr.operations as ops
from ibis.backends.materialize.api import mz_now, mz_top_k
from ibis.backends.postgres import Backend as PostgresBackend
from ibis.backends.sql.compilers.materialize import MaterializeCompiler

__all__ = ("Backend", "mz_now", "mz_top_k")


class Backend(PostgresBackend):
    """Materialize backend for Ibis.

    Materialize is live data layer for apps and agents that allows you to create
    up-to-the-second views into any aspect of your business. It does this by
    maintaining incrementally updated, consistent views over changing data.
    Unlike traditional databases that recompute queries on each execution, Materialize
    continuously updates query results as new data arrives, enabling applications
    to read fresh, consistent results with low latency.

    To learn more about Materialize see: https://materialize.com/docs/
    """

    name = "materialize"
    compiler = MaterializeCompiler()
    supports_python_udfs = False
    supports_temporary_tables = (
        True  # Materialize supports temp tables in mz_temp schema
    )

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 6875,
        database: str | None = None,
        schema: str | None = None,
        autocommit: bool = True,
        cluster: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to Materialize database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number (default: 6875 for Materialize)
        database
            Database to connect to
        schema
            Schema to use. If `None`, uses the default search_path.
        autocommit
            Whether or not to autocommit (default: True)
        cluster
            Default cluster to use for queries. If `None`, uses Materialize's
            default cluster. You can change clusters later with `SET CLUSTER`.
        kwargs
            Additional keyword arguments to pass to the backend client connection.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_MATERIALIZE_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MATERIALIZE_USER", "materialize")
        >>> password = os.environ.get("IBIS_TEST_MATERIALIZE_PASSWORD", "")
        >>> database = os.environ.get("IBIS_TEST_MATERIALIZE_DATABASE", "materialize")
        >>> con = ibis.materialize.connect(
        ...     database=database, host=host, user=user, password=password
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]

        Connect with a specific cluster:

        >>> con = ibis.materialize.connect(
        ...     database="materialize", host="localhost", user="materialize", cluster="quickstart"
        ... )
        """
        import psycopg

        self.con = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database,
            autocommit=autocommit,
            **kwargs,
        )

        self._post_connect()

        # Set cluster if specified
        if cluster is not None:
            with self.begin() as cur:
                cur.execute(f"SET cluster = '{cluster}'")

    @property
    def _session_temp_db(self) -> str:
        """Return the Materialize temporary schema name.

        Materialize stores temporary tables in the mz_temp schema,
        unlike PostgreSQL which uses pg_temp_N schemas.
        """
        return "mz_temp"

    @property
    def version(self):
        """Get Materialize version.

        Returns the version string with leading 'v' stripped for consistency.
        E.g., "v0.158.2" becomes "0.158.2".
        """
        # Materialize has mz_version() function
        try:
            with self.begin() as cur:
                cur.execute("SELECT mz_version()")
                result = cur.fetchone()
                if result and result[0]:
                    version_str = result[0]
                    # Strip leading 'v' if present
                    return (
                        version_str.lstrip("v")
                        if version_str.startswith("v")
                        else version_str
                    )
                return "unknown"
        except Exception:  # noqa: BLE001
            # Fallback to server_version if mz_version() doesn't work
            return super().version

    @property
    def current_cluster(self) -> str:
        """Get the currently active cluster for this session.

        Returns
        -------
        str
            Name of the currently active cluster

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.current_cluster
        'quickstart'

        See Also
        --------
        set_cluster : Switch to a different cluster
        list_clusters : List all available clusters
        """
        with self.begin() as cur:
            cur.execute("SHOW cluster")
            result = cur.fetchone()
            return result[0] if result else None

    def set_cluster(self, name: str) -> None:
        """Set the active cluster for this session.

        This changes which cluster will be used for subsequent queries,
        materialized views, indexes, and other compute operations.

        Parameters
        ----------
        name
            Name of the cluster to switch to

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.set_cluster("production")
        >>> con.current_cluster
        'production'

        Switch clusters for different workloads:

        >>> # Use analytics cluster for heavy queries
        >>> con.set_cluster("analytics")
        >>> result = con.table("large_dataset").aggregate(...)
        >>>
        >>> # Switch back to default for light queries
        >>> con.set_cluster("quickstart")

        See Also
        --------
        current_cluster : Get the currently active cluster
        list_clusters : List all available clusters
        create_cluster : Create a new cluster
        """
        with self.begin() as cur:
            # Use quoted identifier to prevent SQL injection
            quoted_name = sg.to_identifier(name, quoted=True).sql(self.dialect)
            cur.execute(f"SET cluster = {quoted_name}")

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """Register an in-memory table using COPY FROM STDIN.

        Materialize cannot mix DDL and DML in transactions, so we:
        1. CREATE TABLE outside transaction
        2. Use COPY FROM STDIN to load data (separate transaction)

        This approach is ~6x faster than executemany INSERT.
        """
        import ibis.common.exceptions as exc

        schema = op.schema
        if null_columns := schema.null_fields:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted
        type_mapper = self.compiler.type_mapper

        # Use the compiler's type_mapper to convert types (MaterializeType)
        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(col_name, quoted=quoted),
                kind=type_mapper.from_ibis(typ),
                constraints=None if typ.nullable else [sge.NotNullColumnConstraint()],
            )
            for col_name, typ in schema.items()
        ]

        # 1. CREATE TABLE (first transaction)
        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=column_defs,
            ),
            properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
        )
        create_stmt_sql = create_stmt.sql(self.dialect)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_stmt_sql)
        con.commit()

        # 2. Load data (second transaction)
        table = op.data.to_pyarrow(schema)

        # Check if schema contains complex types that CSV doesn't support
        has_complex_types = any(dt.is_nested() for dt in schema.types)

        if has_complex_types:
            # Use INSERT statements for complex types (arrays, maps, structs)
            # PyArrow CSV writer doesn't support these types
            import numpy as np
            import pandas as pd

            df = table.to_pandas()

            def clean_value(val):
                """Convert pandas/numpy values to Python native types for psycopg."""
                if val is None:
                    return None
                elif isinstance(val, np.ndarray):
                    # Recursively clean nested arrays
                    return [clean_value(v) for v in val.tolist()]
                elif isinstance(val, (list, tuple)):
                    # Recursively clean list elements
                    return [clean_value(v) for v in val]
                elif isinstance(val, dict):
                    return val
                elif isinstance(val, (np.integer, np.floating)):
                    # Convert numpy scalar to Python native
                    return val.item()
                elif not isinstance(val, (str, dict)) and pd.isna(val):
                    return None
                else:
                    return val

            if len(df) > 0:
                from psycopg import sql

                columns = list(schema.keys())

                # Build placeholders with explicit type casting for correct type inference
                import re

                placeholders = []
                for col_name in columns:
                    ibis_type = schema[col_name]
                    sql_type = self.compiler.type_mapper.to_string(ibis_type)
                    # Remove dimension specifications to avoid cardinality constraints
                    sql_type = re.sub(r"\[\d+\]", "[]", sql_type)
                    # Add explicit cast - using sql.SQL for safe composition
                    placeholders.append(sql.SQL("%s::{}").format(sql.SQL(sql_type)))

                # Use psycopg's sql module for safe SQL composition
                insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(name),
                    sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                    sql.SQL(", ").join(placeholders),
                )

                # Build list of row values for executemany
                batch_size = 1000
                with con.cursor() as cursor:
                    for i in range(0, len(df), batch_size):
                        batch = df.iloc[i : i + batch_size]
                        rows = []

                        for _, row in batch.iterrows():
                            row_values = tuple(
                                clean_value(row[col_name]) for col_name in columns
                            )
                            rows.append(row_values)

                        # Use executemany for efficient batch insert
                        cursor.executemany(insert_sql, rows)
                con.commit()
        else:
            # Use efficient CSV loading for simple types
            import io

            import pyarrow.csv as pacsv

            # Write PyArrow table to CSV in memory
            csv_buffer = io.BytesIO()
            pacsv.write_csv(table, csv_buffer)
            csv_buffer.seek(0)

            # Use COPY FROM STDIN to load data
            columns = list(schema.keys())
            col_list = ", ".join(f'"{c}"' for c in columns)
            copy_sql = (
                f'COPY "{name}" ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER true)'
            )

            with con.cursor() as cur:
                with cur.copy(copy_sql) as copy:
                    while data := csv_buffer.read(8192):
                        copy.write(data)
            con.commit()

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ):
        """Get the schema for a table, view, or materialized view."""
        import ibis.common.exceptions as com
        import ibis.expr.schema as sch

        type_mapper = self.compiler.type_mapper
        con = self.con
        params = {"name": name}

        # Try mz_* catalogs first (works for both temp and regular objects)
        mz_query = """\
SELECT
  c.name AS column_name,
  c.type AS data_type,
  c.nullable AS nullable
FROM mz_columns c
LEFT JOIN mz_tables t ON c.id = t.id
LEFT JOIN mz_views v ON c.id = v.id
LEFT JOIN mz_materialized_views mv ON c.id = mv.id
WHERE COALESCE(t.name, v.name, mv.name) = %(name)s
ORDER BY c.position ASC"""

        with con.cursor() as cursor:
            rows = cursor.execute(mz_query, params).fetchall()

        # If found in mz_* catalogs, return schema
        if rows:
            return sch.Schema(
                {
                    col: type_mapper.from_string(typestr, nullable=nullable)
                    for col, typestr, nullable in rows
                }
            )

        # Fallback to pg_catalog for system tables
        dbs = [database or self.current_database]
        schema_conditions = " OR ".join([f"n.nspname = '{db}'" for db in dbs])

        # S608 is a false positive - db comes from self.current_database, not user input
        pg_query = f"""\
SELECT
  a.attname AS column_name,
  CASE
    WHEN EXISTS(
      SELECT 1
      FROM pg_catalog.pg_type t
      INNER JOIN pg_catalog.pg_enum e
              ON e.enumtypid = t.oid
             AND t.typname = pg_catalog.format_type(a.atttypid, a.atttypmod)
    ) THEN 'enum'
    ELSE pg_catalog.format_type(a.atttypid, a.atttypmod)
  END AS data_type,
  NOT a.attnotnull AS nullable
FROM pg_catalog.pg_attribute a
INNER JOIN pg_catalog.pg_class c
   ON a.attrelid = c.oid
INNER JOIN pg_catalog.pg_namespace n
   ON c.relnamespace = n.oid
WHERE a.attnum > 0
  AND NOT a.attisdropped
  AND ({schema_conditions})
  AND c.relname = %(name)s
ORDER BY a.attnum ASC"""  # noqa: S608

        with con.cursor() as cursor:
            rows = cursor.execute(pg_query, params).fetchall()

        if not rows:
            raise com.TableNotFound(name)

        return sch.Schema(
            {
                col: type_mapper.from_string(typestr, nullable=nullable)
                for col, typestr, nullable in rows
            }
        )

    def create_table(
        self,
        name: str,
        /,
        obj=None,
        *,
        schema=None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in Materialize."""

        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ibis.expr.types.Table):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)
            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite:
            temp_name = ibis.util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect
        type_mapper = self.compiler.type_mapper

        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        table_expr = sg.table(temp_name, catalog=database, quoted=quoted)

        # Use the compiler's type_mapper to convert types (MaterializeType)
        # instead of schema.to_sqlglot_column_defs which uses default Postgres types
        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(name, quoted=quoted),
                kind=type_mapper.from_ibis(typ),
                constraints=None if typ.nullable else [sge.NotNullColumnConstraint()],
            )
            for name, typ in schema.items()
        ]

        target = sge.Schema(this=table_expr, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        ).sql(dialect)

        this = sg.table(name, catalog=database, quoted=quoted)
        this_no_catalog = sg.table(name, quoted=quoted)

        con = self.con

        # Execute CREATE TABLE (no transaction)
        with con.cursor() as cursor:
            cursor.execute(create_stmt)
        con.commit()

        # Execute INSERT if needed (separate transaction)
        if query is not None:
            insert_stmt = sge.Insert(this=table_expr, expression=query).sql(dialect)
            with con.cursor() as cursor:
                cursor.execute(insert_stmt)
            con.commit()

        # Handle overwrite with RENAME (separate transaction)
        if overwrite:
            drop_stmt = sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect)
            rename_stmt = f"ALTER TABLE IF EXISTS {table_expr.sql(dialect)} RENAME TO {this_no_catalog.sql(dialect)}"

            with con.cursor() as cursor:
                cursor.execute(drop_stmt)
            con.commit()

            with con.cursor() as cursor:
                cursor.execute(rename_stmt)
            con.commit()

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def create_materialized_view(
        self,
        name: str,
        /,
        obj: ibis.expr.types.Table,
        *,
        database: str | None = None,
        schema: str | None = None,
        overwrite: bool = False,
    ) -> ibis.expr.types.Table:
        """Create a materialized view.

        Materialized views that maintains fresh results by incrementally updating them as new data arrives.
        They are particularly useful when you need cross-cluster access to results or want to sink data to
        external systems like Kafka. When you create a materialized view, you specify a cluster responsible
        for maintaining it, but the results can be queried from any cluster. This allows you to separate the
        compute resources used for view maintenance from those used for serving queries.

        If you do not need cross-cluster sharing, and you are primarily interested in fast query performance
        within a single cluster, you may prefer to create a view and index it. In Materialize, indexes on views
        also maintain results incrementally, but store them in memory, scoped to the cluster where the index was
        created. This approach offers lower latency for direct querying within that cluster.

        Parameters
        ----------
        name
            Materialized view name to create.
        obj
            The select statement to materialize.
        database
            Name of the database (catalog) where the view will be created.
        schema
            Name of the schema where the view will be created.
        overwrite
            Whether to overwrite the existing materialized view with the same name.
            Uses CREATE OR REPLACE syntax.

        Returns
        -------
        Table
            Table expression representing the materialized view

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> table = con.table("orders")
        >>> daily_summary = table.group_by("date").aggregate(
        ...     total=table.amount.sum(), count=table.count()
        ... )
        >>> mv = con.create_materialized_view("daily_orders", daily_summary)
        """
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        table = sg.table(name, catalog=database, db=schema, quoted=self.compiler.quoted)

        create_stmt = sge.Create(
            this=table,
            kind="MATERIALIZED VIEW",
            expression=self.compile(obj),
            replace=overwrite,  # Use CREATE OR REPLACE when overwrite=True
        )

        self._run_pre_execute_hooks(obj)

        con = self.con
        # Execute CREATE [OR REPLACE] MATERIALIZED VIEW
        with con.cursor() as cursor:
            cursor.execute(create_stmt.sql(self.dialect))
        con.commit()

        return self.table(name, database=database)

    def drop_materialized_view(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        """Drop a materialized view.

        Parameters
        ----------
        name
            Materialized view name to drop.
        database
            Name of the database (catalog) where the view exists, if not the default.
        schema
            Name of the schema where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        cascade
            If `True`, also drop dependent objects (views, indexes, etc.).

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_materialized_view("daily_orders")
        >>> con.drop_materialized_view("old_view", force=True)  # Won't error if missing
        >>> con.drop_materialized_view("old_view", force=True, cascade=True)  # Drop with dependents
        """
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        drop_stmt = sge.Drop(
            this=sg.table(
                name, catalog=database, db=schema, quoted=self.compiler.quoted
            ),
            kind="MATERIALIZED VIEW",
            exists=force,
            cascade=cascade,
        )

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(drop_stmt.sql(self.dialect))
        con.commit()

    def list_materialized_views(
        self,
        *,
        database: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List materialized views in Materialize.

        Parameters
        ----------
        database
            Database/schema to list materialized views from.
            If None, uses current database.
        like
            Pattern to filter materialized view names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of materialized view names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_materialized_views()
        ['daily_orders', 'weekly_summary', 'user_stats']
        >>> con.list_materialized_views(like="daily%")
        ['daily_orders']
        """
        query = """
        SELECT mv.name
        FROM mz_catalog.mz_materialized_views mv
        JOIN mz_catalog.mz_schemas s ON mv.schema_id = s.id
        WHERE s.name = %(database)s
        """

        params = {"database": database or self.current_database}

        if like is not None:
            query += " AND mv.name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY mv.name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def create_source(
        self,
        name: str,
        /,
        *,
        source_schema: ibis.Schema | None = None,
        database: str | None = None,
        schema: str | None = None,
        connection: str | None = None,
        connector: str | None = None,
        properties: dict[str, str] | None = None,
        format_spec: dict[str, str] | None = None,
        envelope: str | None = None,
        include_properties: list[str] | None = None,
        for_all_tables: bool = False,
        for_schemas: list[str] | None = None,
        for_tables: list[tuple[str, str]] | None = None,
    ) -> ibis.expr.types.Table | None:
        """Create a source in Materialize.

        This method supports creating sources from various systems including:
        - Load generators (COUNTER, AUCTION, TPCH, MARKETING, KEY VALUE)
        - Kafka/Redpanda message brokers
        - PostgreSQL, MySQL, SQL Server (CDC)
        - Webhooks

        The API is designed for compatibility with RisingWave's create_source while
        supporting Materialize-specific features.

        Parameters
        ----------
        name
            Source name to create.
        source_schema
            Ibis schema defining the structure of data from the source (columns and types).
            Required for some Kafka sources to specify the shape of incoming data.
        database
            Name of the database (catalog) where the source will be created.
        schema
            Name of the schema where the source will be created.
        connection
            Name of the connection object (for Kafka, Postgres, MySQL, etc.).
            Must be created beforehand using CREATE CONNECTION.
        connector
            Type of connector: 'COUNTER', 'AUCTION', 'KAFKA', 'POSTGRES', etc.
            Load generator types (COUNTER, AUCTION, etc.) are detected automatically.
        properties
            Connector-specific properties, e.g.:
            - Kafka: {'TOPIC': 'my_topic'}
            - Postgres: {'PUBLICATION': 'my_pub'}
            - Load Generator: {'TICK INTERVAL': '1s', 'SCALE FACTOR': '0.01'}
        format_spec
            Format specifications, e.g.:
            {'KEY FORMAT': 'JSON', 'VALUE FORMAT': 'JSON'}
            or {'FORMAT': 'JSON'} for non-Kafka sources
        envelope
            Data envelope type: 'NONE', 'UPSERT', or 'DEBEZIUM'
        include_properties
            List of metadata to include, e.g., ['KEY', 'PARTITION', 'OFFSET']
        for_all_tables
            Create subsources for all tables (Postgres/MySQL) or all load generator tables
        for_schemas
            List of schemas to create subsources for (Postgres/MySQL)
        for_tables
            List of (table_name, subsource_name) tuples

        Returns
        -------
        Table | None
            Table expression for the source. Returns None for multi-table sources
            (when for_all_tables=True).

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        >>> # Load generator
        >>> counter = con.create_source(
        ...     "my_counter", connector="COUNTER", properties={"TICK INTERVAL": "500ms"}
        ... )

        >>> # Kafka source
        >>> kafka_src = con.create_source(
        ...     "kafka_data",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "my_topic"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ... )

        >>> # PostgreSQL CDC
        >>> pg_src = con.create_source(
        ...     "pg_tables",
        ...     connector="POSTGRES",
        ...     connection="pg_conn",
        ...     properties={"PUBLICATION": "mz_source"},
        ...     for_all_tables=True,
        ... )
        """
        # Validate parameters
        if connector is None and connection is None:
            raise ValueError("Must specify either connector or connection")

        # Load generator types
        load_generator_types = {
            "COUNTER",
            "AUCTION",
            "TPCH",
            "MARKETING",
            "KEY VALUE",
            "DATUMS",
        }

        # Detect if connector is a load generator
        is_load_generator = connector and connector.upper() in load_generator_types

        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        source_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        # Build CREATE SOURCE statement
        create_parts = [f"CREATE SOURCE {source_table.sql(self.dialect)}"]

        # Build FROM clause based on connector type
        if is_load_generator:
            # Load generator source
            gen_type = connector.upper()
            create_parts.append(f"FROM LOAD GENERATOR {gen_type}")

            if properties:
                opts_str = ", ".join(
                    f"{key} '{value}'" if " " in key else f"{key} {value}"
                    for key, value in properties.items()
                )
                create_parts.append(f"({opts_str})")

        elif connection is not None:
            # Connection-based source (Kafka, Postgres, etc.)
            # Add source_schema if provided
            if source_schema is not None:
                # Use the compiler's type_mapper to convert types (MaterializeType)
                type_mapper = self.compiler.type_mapper
                schema_cols = [
                    sge.ColumnDef(
                        this=sg.to_identifier(col_name, quoted=quoted),
                        kind=type_mapper.from_ibis(typ),
                        constraints=None
                        if typ.nullable
                        else [sge.NotNullColumnConstraint()],
                    )
                    for col_name, typ in source_schema.items()
                ]
                col_defs = ", ".join(col.sql(self.dialect) for col in schema_cols)
                create_parts.append(f"({col_defs})")

            # Determine connector type from explicit parameter or default to Kafka
            connector_type = connector.upper() if connector else "KAFKA"
            create_parts.append(f"FROM {connector_type} CONNECTION {connection}")

            # Add properties in parentheses
            if properties:
                props_str = ", ".join(
                    f"{key} '{value}'" for key, value in properties.items()
                )
                create_parts.append(f"({props_str})")

        # Add format specifications
        if format_spec:
            for key, value in format_spec.items():
                create_parts.append(f"{key} {value}")

        # Add envelope
        if envelope:
            create_parts.append(f"ENVELOPE {envelope}")

        # Add INCLUDE clauses
        if include_properties:
            create_parts.extend(f"INCLUDE {prop}" for prop in include_properties)

        # Add FOR clauses
        if for_all_tables:
            create_parts.append("FOR ALL TABLES")
        elif for_schemas:
            schemas_str = ", ".join(f"'{schema}'" for schema in for_schemas)
            create_parts.append(f"FOR SCHEMAS ({schemas_str})")
        elif for_tables:
            tables_str = ", ".join(f"{table} AS {alias}" for table, alias in for_tables)
            create_parts.append(f"FOR TABLES ({tables_str})")

        create_sql = " ".join(create_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_sql)
        con.commit()

        # Return None for multi-table sources
        if for_all_tables or for_schemas or for_tables:
            return None
        return self.table(name, database=database)

    def drop_source(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        """Drop a source.

        Parameters
        ----------
        name
            Source name to drop.
        database
            Name of the database (catalog) where the source exists, if not the default.
        schema
            Name of the schema where the source exists, if not the default.
        force
            If `False`, an exception is raised if the source does not exist.
        cascade
            If `True`, also drops dependent objects (views, materialized views).

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_source("my_counter")
        >>> con.drop_source("old_source", force=True, cascade=True)
        """
        drop_stmt_parts = ["DROP SOURCE"]

        if force:
            drop_stmt_parts.append("IF EXISTS")

        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        source_table = sg.table(name, catalog=database, db=schema, quoted=quoted)
        drop_stmt_parts.append(source_table.sql(self.dialect))

        if cascade:
            drop_stmt_parts.append("CASCADE")

        drop_sql = " ".join(drop_stmt_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(drop_sql)
        con.commit()

    def list_sources(
        self,
        *,
        database: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List sources in Materialize.

        Parameters
        ----------
        database
            Database/schema to list sources from.
            If None, uses current database.
        like
            Pattern to filter source names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of source names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_sources()
        ['my_counter', 'auction_house', 'kafka_source']
        >>> con.list_sources(like="auction%")
        ['auction_house']
        """
        query = """
        SELECT s.name
        FROM mz_catalog.mz_sources s
        JOIN mz_catalog.mz_schemas sc ON s.schema_id = sc.id
        WHERE sc.name = %(database)s
        """

        params = {"database": database or self.current_database}

        if like is not None:
            query += " AND s.name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY s.name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def subscribe(
        self,
        obj: str | ibis.expr.types.Table,
        /,
        *,
        envelope: str | None = None,
        snapshot: bool = True,
        as_of: int | None = None,
        up_to: int | None = None,
        progress: bool = False,
        batch_size: int = 1000,
        format: str = "pandas",
    ):
        """Subscribe to real-time changes in a table, view, or materialized view.

        SUBSCRIBE enables streaming change data capture (CDC) from Materialize
        relations. Unlike regular queries that return a snapshot, SUBSCRIBE
        continuously streams updates as they happen, making it ideal for:

        - Real-time dashboards and monitoring
        - Event-driven architectures and triggers
        - Syncing data to external systems
        - Live data pipelines

        The stream continues indefinitely (unless `up_to` is specified) and
        delivers changes incrementally as pandas DataFrames.

        Parameters
        ----------
        obj
            Name of source/table/view/materialized view, or an Ibis table
            expression to subscribe to.
        envelope
            Output format: 'UPSERT' or 'DEBEZIUM'. If None, uses default format
            with mz_diff column.
        snapshot
            If True (default), emits the initial state before streaming changes.
            If False, only emits changes that occur after subscription starts.
        as_of
            Start streaming from this Materialize timestamp.
        up_to
            Stop streaming at this Materialize timestamp (for time-travel queries).
        progress
            If True, emits progress updates in addition to data changes.
        batch_size
            Number of rows to fetch per batch (default: 1000).
        format
            Output format for batches: 'pandas' (default), 'arrow', or 'polars'.
            - 'pandas': Returns pandas DataFrames (familiar, feature-rich)
            - 'arrow': Returns PyArrow RecordBatches (efficient, zero-copy)
            - 'polars': Returns Polars DataFrames (fast, modern API)

        Returns
        -------
        Iterator[pd.DataFrame | pa.RecordBatch | pl.DataFrame]
            Generator that yields batches of changes. Format depends on `format` parameter. Each batch includes:

            - **mz_timestamp**: Materialize's logical timestamp for this change
            - **mz_diff**: Change type indicator:
                - `1` = row inserted (or new version after update)
                - `-1` = row deleted (or old version before update)
                - `0` = progress message (only if progress=True)
            - All columns from the subscribed relation

            **Important**: Row updates appear as a *delete* (-1) followed by an
            *insert* (+1). Filter for `mz_diff == 1` to see only current/new rows.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        **Simplest example** - Stream all changes:

        >>> for batch in con.subscribe("orders"):
        ...     print(f"Received {len(batch)} changes")
        ...     print(batch)

        **Filter for inserts only** (ignore deletes and old update versions):

        >>> for batch in con.subscribe("orders"):
        ...     new_rows = batch[batch["mz_diff"] == 1]
        ...     for _, row in new_rows.iterrows():
        ...         print(f"New order: {row['order_id']}")

        **Subscribe to a live query** (not just a table name):

        >>> high_value = con.table("orders").filter(_.amount > 10000)
        >>> for batch in con.subscribe(high_value):
        ...     send_alert(batch)

        **Stream only new changes** (skip initial snapshot):

        >>> for batch in con.subscribe("events", snapshot=False):
        ...     # Only see changes after subscription starts
        ...     process_new_events(batch)

        **Time-bounded streaming** (useful for testing):

        >>> for batch in con.subscribe("sensors", up_to=end_timestamp):
        ...     analyze_batch(batch)

        **Using UPSERT envelope** (for syncing to databases):

        >>> for batch in con.subscribe("customers", envelope="UPSERT"):
        ...     # batch has 'before' and 'after' columns
        ...     sync_to_warehouse(batch)

        **Using Arrow format** (efficient, lower memory overhead):

        >>> for batch in con.subscribe("events", format="arrow"):
        ...     # batch is pyarrow.RecordBatch
        ...     # Better for high-throughput or memory-constrained scenarios
        ...     process_arrow_batch(batch)

        **Using Polars format** (fast, modern DataFrame):

        >>> for batch in con.subscribe("logs", format="polars"):
        ...     # batch is polars.DataFrame
        ...     # Fast filtering: batch.filter(pl.col("mz_diff") == 1)
        ...     analyze_with_polars(batch)

        Notes
        -----
        - The stream runs indefinitely unless `up_to` is specified
        - Use Ctrl+C or `break` to stop streaming
        - SUBSCRIBE keeps a connection open; use it in a try/finally block
        - Changes are ordered by mz_timestamp within each batch
        - Requires SELECT privilege on the relation
        - See Materialize SUBSCRIBE docs for more details on envelopes and
          progress messages

        See Also
        --------
        create_materialized_view : Create views that update incrementally
        create_source : Ingest streaming data from external systems

        """
        # Validate format parameter
        if format not in ("pandas", "arrow", "polars"):
            raise ValueError(
                f"format must be 'pandas', 'arrow', or 'polars', got {format!r}"
            )

        # Import appropriate libraries based on format
        if format == "pandas":
            import pandas as pd
        elif format == "arrow":
            import pyarrow as pa
        elif format == "polars":
            try:
                import polars as pl
            except ImportError:
                raise ImportError(
                    "polars is required for format='polars'. "
                    "Install it with: pip install polars"
                ) from None

        # Build SUBSCRIBE SQL
        if isinstance(obj, str):
            sql_parts = [f"SUBSCRIBE {obj}"]
        else:
            # It's an Ibis expression
            self._run_pre_execute_hooks(obj)
            query = self.compiler.to_sqlglot(obj)
            sql_parts = [f"SUBSCRIBE ({query.sql(self.dialect)})"]

        # Add envelope if specified
        if envelope is not None:
            envelope_upper = envelope.upper()
            if envelope_upper not in ("UPSERT", "DEBEZIUM"):
                raise ValueError("envelope must be 'UPSERT' or 'DEBEZIUM'")
            sql_parts.append(f"ENVELOPE {envelope_upper}")

        # Build WITH options
        options = []
        if not snapshot:
            options.append("SNAPSHOT = FALSE")
        if progress:
            options.append("PROGRESS = TRUE")
        if options:
            sql_parts.append(f"WITH ({', '.join(options)})")

        # Add AS OF timestamp
        if as_of is not None:
            sql_parts.append(f"AS OF {as_of}")

        # Add UP TO timestamp
        if up_to is not None:
            sql_parts.append(f"UP TO {up_to}")

        sql = " ".join(sql_parts)

        # SUBSCRIBE needs a dedicated connection since it blocks
        # Create a new connection with the same parameters
        import psycopg

        con_params = self.con.info.get_parameters()
        sub_con = psycopg.connect(
            host=con_params.get("host", "localhost"),
            port=int(con_params.get("port", 6875)),
            dbname=con_params.get("dbname", "materialize"),
            user=con_params.get("user"),
            password=con_params.get("password")
            if hasattr(self.con, "_password")
            else None,
            autocommit=False,  # Need transaction for DECLARE CURSOR
        )

        cursor = sub_con.cursor()

        try:
            # Begin transaction and declare cursor for subscription
            cursor.execute("BEGIN")
            cursor_name = "ibis_sub_cursor"
            cursor.execute(f"DECLARE {cursor_name} CURSOR FOR {sql}")

            # Get column names after declaring cursor
            columns = None

            # Fetch results in batches
            while True:
                # Fetch a batch of rows
                cursor.execute(f"FETCH {batch_size} {cursor_name}")
                rows = cursor.fetchall()

                # Get columns from first fetch
                if columns is None and cursor.description:
                    columns = [desc[0] for desc in cursor.description]

                # If no rows and up_to was specified, we're done
                if not rows:
                    if up_to is not None:
                        break
                    # Otherwise wait a bit and try again
                    continue

                # Yield batch in requested format
                if columns:
                    if format == "pandas":
                        import pandas as pd

                        yield pd.DataFrame(rows, columns=columns)
                    elif format == "arrow":
                        import pyarrow as pa

                        # Convert rows (list of tuples) to columnar format
                        # PyArrow expects columnar data (list per column)
                        arrays = []
                        for col_idx in range(len(columns)):
                            col_data = [row[col_idx] for row in rows]
                            arrays.append(pa.array(col_data))

                        # Create RecordBatch with column names
                        batch = pa.RecordBatch.from_arrays(arrays, names=columns)
                        yield batch
                    elif format == "polars":
                        # polars was already imported and validated at method start
                        # Convert rows to dict format (columnar)
                        data = {
                            col: [row[i] for row in rows]
                            for i, col in enumerate(columns)
                        }
                        yield pl.DataFrame(data)

        except (KeyboardInterrupt, GeneratorExit):
            # Allow graceful termination
            pass
        finally:
            # Ensure cursor and connection are closed
            from contextlib import suppress

            with suppress(Exception):
                cursor.close()
            with suppress(Exception):
                sub_con.close()

    def alter_source(
        self,
        name: str,
        /,
        *,
        add_subsources: list[tuple[str, str]] | list[str] | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Alter a source.

        Parameters
        ----------
        name
            Source name to alter.
        add_subsources
            Tables to add as subsources. Can be:
            - List of table names: ['table1', 'table2']
            - List of (table_name, subsource_name) tuples: [('table1', 'sub1')]
        database
            Name of the database (catalog) where the source exists.
        schema
            Name of the schema where the source exists.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Add subsources with default names:

        >>> con.alter_source("pg_source", add_subsources=["orders", "customers"])

        Add subsources with custom names:

        >>> con.alter_source("pg_source", add_subsources=[("orders", "orders_subsource")])

        Add multiple subsources with mixed naming:

        >>> con.alter_source("mysql_source", add_subsources=["table_a", ("table_b", "b")])

        Notes
        -----
        - Only works with PostgreSQL and MySQL sources.
        - Cannot drop subsources (except the progress subsource).
        - Tables must exist in the upstream database.
        - Requires ownership of the source.

        """
        if not add_subsources:
            raise ValueError("Must specify add_subsources")

        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        source_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        # Build subsource specifications
        subsource_specs = []
        for item in add_subsources:
            if isinstance(item, tuple):
                table_name, subsource_name = item
                subsource_specs.append(f"{table_name} AS {subsource_name}")
            else:
                # Just a table name
                subsource_specs.append(item)

        subsources_str = ", ".join(subsource_specs)
        sql = f"ALTER SOURCE {source_table.sql(self.dialect)} ADD SUBSOURCE {subsources_str}"

        with self.begin() as cur:
            cur.execute(sql)

    def create_sink(
        self,
        name: str,
        /,
        *,
        sink_from: str | None = None,
        obj: ibis.expr.types.Table | None = None,
        connector: str | None = None,
        connection: str | None = None,
        properties: dict[str, str] | None = None,
        format_spec: dict[str, str] | None = None,
        envelope: str | None = None,
        key: list[str] | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create a sink in Materialize.

        Sinks allow you to stream data from Materialize to external systems.

        Parameters
        ----------
        name
            Sink name to create.
        sink_from
            Name of the table/materialized view/source to sink from.
            Either `sink_from` or `obj` must be specified (RisingWave compatibility).
        obj
            Ibis table expression to sink from. Either `sink_from` or `obj` must be
            specified (RisingWave compatibility).
        connector
            Type of connector: 'KAFKA' (default if connection is provided).
        connection
            Name of the connection object (for Kafka).
            Must be created beforehand using CREATE CONNECTION.
        properties
            Connector-specific properties, e.g.:
            - Kafka: {'TOPIC': 'events'}
        format_spec
            Format specifications. Can specify either:
            - Single format: {'FORMAT': 'JSON'}
            - Key/Value formats: {'KEY FORMAT': 'TEXT', 'VALUE FORMAT': 'JSON'}
        envelope
            Data envelope type: 'UPSERT' or 'DEBEZIUM'
        key
            List of column names to use as the message key.
            Required for UPSERT envelope.
        database
            Name of the database (catalog) where the sink will be created.
        schema
            Name of the schema where the sink will be created.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        >>> # Create sink from materialized view
        >>> con.create_sink(
        ...     "kafka_sink",
        ...     sink_from="my_materialized_view",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "events"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ...     key=["id"],
        ... )

        >>> # Create sink from expression (RisingWave style)
        >>> expr = con.table("orders").filter(orders.status == "complete")
        >>> con.create_sink(
        ...     "completed_orders_sink",
        ...     obj=expr,
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "completed_orders"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ...     key=["order_id"],
        ... )
        """
        # Validate parameters
        if sink_from is None and obj is None:
            raise ValueError("Either `sink_from` or `obj` must be specified")
        if sink_from is not None and obj is not None:
            raise ValueError("Only one of `sink_from` or `obj` can be specified")

        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        sink_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        # Build CREATE SINK statement
        create_parts = [f"CREATE SINK {sink_table.sql(self.dialect)}"]

        # Add FROM clause
        if sink_from is not None:
            # Direct table/view reference
            from_table = sg.table(sink_from, quoted=quoted)
            create_parts.append(f"FROM {from_table.sql(self.dialect)}")
        else:
            # Expression - need to compile it
            self._run_pre_execute_hooks(obj)
            query = self.compiler.to_sqlglot(obj)
            create_parts.append(f"FROM ({query.sql(self.dialect)})")

        # Determine connector type
        connector_type = connector.upper() if connector else "KAFKA"

        # Add INTO clause
        if connection is None:
            raise ValueError("connection parameter is required")

        create_parts.append(f"INTO {connector_type} CONNECTION {connection}")

        # Add properties in parentheses
        if properties:
            props_str = ", ".join(
                f"{key} '{value}'" for key, value in properties.items()
            )
            create_parts.append(f"({props_str})")

        # Add KEY clause
        if key:
            key_cols = ", ".join(key)
            create_parts.append(f"KEY ({key_cols})")

        # Add format specifications
        if format_spec:
            for key_name, value in format_spec.items():
                create_parts.append(f"{key_name} {value}")

        # Add envelope
        if envelope:
            create_parts.append(f"ENVELOPE {envelope}")

        create_sql = " ".join(create_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_sql)
        con.commit()

    def drop_sink(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a sink.

        Parameters
        ----------
        name
            Sink name to drop.
        database
            Name of the database (catalog) where the sink exists, if not the default.
        schema
            Name of the schema where the sink exists, if not the default.
        force
            If `False`, an exception is raised if the sink does not exist.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_sink("my_sink", force=True)
        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        sink_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        drop_stmt = sge.Drop(
            this=sink_table,
            kind="SINK",
            exists=force,
        )

        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def list_sinks(
        self,
        *,
        database: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List sinks in Materialize.

        Parameters
        ----------
        database
            Database/schema to list sinks from.
            If None, uses current database.
        like
            Pattern to filter sink names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of sink names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_sinks()
        ['kafka_sink', 'orders_sink']
        >>> con.list_sinks(like="kafka%")
        ['kafka_sink']
        """
        query = """
        SELECT s.name
        FROM mz_catalog.mz_sinks s
        JOIN mz_catalog.mz_schemas sc ON s.schema_id = sc.id
        WHERE sc.name = %(database)s
        """

        params = {"database": database or self.current_database}

        if like is not None:
            query += " AND s.name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY s.name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def alter_sink(
        self,
        name: str,
        /,
        *,
        set_from: str,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Alter a sink to read from a different upstream relation.

        Allows cutting a sink over to a new upstream relation (table, view, or
        materialized view) without disrupting downstream consumers. Useful for
        blue/green deployments.

        Parameters
        ----------
        name
            Sink name to alter.
        set_from
            Name of the new upstream relation (table/view/materialized view) to
            read from. The new relation must be compatible with the original sink
            definition.
        database
            Name of the database (catalog) where the sink exists.
        schema
            Name of the schema where the sink exists.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Cut over to a new materialized view:

        >>> con.alter_sink("orders_sink", set_from="orders_v2")

        Cut over to a new table:

        >>> con.alter_sink("avro_sink", set_from="matview_new")

        Notes
        -----
        - The new relation must have a compatible schema
        - For Avro sinks, the schema must be compatible with previously published
          schema
        - Materialize only emits updates occurring after the cutover timestamp
        - Requires SELECT privileges on the new relation
        - Consider potential issues with missing keys or stale values during
          cutover

        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        sink_table = sg.table(name, catalog=database, db=schema, quoted=quoted)
        from_table = sg.table(set_from, quoted=quoted)

        sql = f"ALTER SINK {sink_table.sql(self.dialect)} SET FROM {from_table.sql(self.dialect)}"

        with self.begin() as cur:
            cur.execute(sql)

    def create_connection(
        self,
        name: str,
        /,
        *,
        connection_type: str,
        properties: dict[str, str | Any],
        database: str | None = None,
        schema: str | None = None,
        validate: bool = True,
    ) -> None:
        """Create a connection in Materialize.

        Connections store reusable connection configurations for sources and sinks.
        They enable secure credential management and connection reuse across multiple
        streaming objects.

        Parameters
        ----------
        name
            Connection name to create.
        connection_type
            Type of connection: 'KAFKA', 'POSTGRES', 'MYSQL', 'AWS', 'SSH TUNNEL',
            'AWS PRIVATELINK', 'CONFLUENT SCHEMA REGISTRY'
        properties
            Connection-specific properties as key-value pairs, e.g.:
            - Kafka: {'BROKER': 'localhost:9092', 'SASL MECHANISMS': 'PLAIN', ...}
            - Postgres: {'HOST': 'localhost', 'PORT': '5432', 'DATABASE': 'mydb', ...}
            - AWS: {'REGION': 'us-east-1', 'ACCESS KEY ID': SECRET('aws_key'), ...}
        database
            Name of the database (catalog) where the connection will be created.
        schema
            Name of the schema where the connection will be created.
        validate
            Whether to validate the connection (default: True).
            Set to False to create without validation.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        >>> # Kafka connection with SASL authentication
        >>> con.create_connection(
        ...     "kafka_conn",
        ...     connection_type="KAFKA",
        ...     properties={
        ...         "BROKER": "localhost:9092",
        ...         "SASL MECHANISMS": "PLAIN",
        ...         "SASL USERNAME": "user",
        ...         "SASL PASSWORD": SECRET("kafka_password"),
        ...     },
        ... )

        >>> # PostgreSQL CDC connection
        >>> con.create_connection(
        ...     "pg_conn",
        ...     connection_type="POSTGRES",
        ...     properties={
        ...         "HOST": "localhost",
        ...         "PORT": "5432",
        ...         "DATABASE": "mydb",
        ...         "USER": "postgres",
        ...         "PASSWORD": SECRET("pg_password"),
        ...     },
        ... )

        >>> # AWS connection for S3 sources
        >>> con.create_connection(
        ...     "aws_conn",
        ...     connection_type="AWS",
        ...     properties={
        ...         "REGION": "us-east-1",
        ...         "ACCESS KEY ID": SECRET("aws_key"),
        ...         "SECRET ACCESS KEY": SECRET("aws_secret"),
        ...     },
        ... )
        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        conn_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        # Build CREATE CONNECTION statement
        conn_type_upper = connection_type.upper()
        create_parts = [
            f"CREATE CONNECTION {conn_table.sql(self.dialect)} TO {conn_type_upper}"
        ]

        # Add properties in parentheses
        if properties:
            # Handle both simple values and SECRET() references
            props_list = []
            for key, value in properties.items():
                # Check if value looks like a SECRET() call
                if isinstance(value, str) and value.startswith("SECRET("):
                    # Pass through SECRET() references as-is
                    props_list.append(f"{key} = {value}")
                else:
                    # Quote regular string values
                    props_list.append(f"{key} = '{value}'")

            props_str = ", ".join(props_list)
            create_parts.append(f"({props_str})")

        # Add VALIDATE clause
        if not validate:
            create_parts.append("WITH (VALIDATE = false)")

        create_sql = " ".join(create_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_sql)
        con.commit()

    def drop_connection(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        """Drop a connection.

        Parameters
        ----------
        name
            Connection name to drop.
        database
            Name of the database (catalog) where the connection exists, if not the default.
        schema
            Name of the schema where the connection exists, if not the default.
        force
            If `False`, an exception is raised if the connection does not exist.
        cascade
            If `True`, drop dependent objects (sources, sinks) as well.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_connection("kafka_conn", force=True)
        >>> con.drop_connection("pg_conn", cascade=True)
        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        conn_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        drop_stmt = sge.Drop(
            this=conn_table,
            kind="CONNECTION",
            exists=force,
            cascade=cascade,
        )

        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def list_connections(
        self,
        *,
        database: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List connections in Materialize.

        Parameters
        ----------
        database
            Database/schema to list connections from.
            If None, uses current database.
        like
            Pattern to filter connection names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of connection names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_connections()
        ['kafka_conn', 'pg_conn', 'aws_conn']
        >>> con.list_connections(like="kafka%")
        ['kafka_conn']
        """
        query = """
        SELECT c.name
        FROM mz_catalog.mz_connections c
        JOIN mz_catalog.mz_schemas s ON c.schema_id = s.id
        WHERE s.name = %(database)s
        """

        params = {"database": database or self.current_database}

        if like is not None:
            query += " AND c.name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY c.name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def alter_connection(
        self,
        name: str,
        /,
        *,
        set_options: dict[str, str | Any] | None = None,
        reset_options: list[str] | None = None,
        rotate_keys: bool = False,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Alter a connection's configuration.

        Modify connection parameters, reset them to defaults, or rotate SSH tunnel
        keys.

        Parameters
        ----------
        name
            Connection name to alter.
        set_options
            Dictionary of connection options to set. Options depend on connection
            type:
            - Kafka: 'BROKER', 'SASL MECHANISMS', etc.
            - Postgres: 'HOST', 'PORT', 'DATABASE', etc.
            - AWS: 'REGION', 'ACCESS KEY ID', etc.
            Values can be strings or SECRET() references.
        reset_options
            List of option names to reset to defaults.
        rotate_keys
            If True, rotate SSH tunnel key pairs. Only valid for SSH TUNNEL
            connections. Requires manual update of SSH bastion server keys.
        database
            Name of the database (catalog) where the connection exists.
        schema
            Name of the schema where the connection exists.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Update connection broker:

        >>> con.alter_connection("kafka_conn", set_options={"BROKER": "new-broker:9092"})

        Reset connection port to default:

        >>> con.alter_connection("pg_conn", reset_options=["PORT"])

        Rotate SSH tunnel keys:

        >>> con.alter_connection("ssh_conn", rotate_keys=True)

        Notes
        -----
        - Changes are applied atomically
        - Cannot modify the same parameter via both SET and RESET
        - For SSH key rotation, update bastion server keys manually after rotation

        """
        if not any([set_options, reset_options, rotate_keys]):
            raise ValueError(
                "Must specify at least one of: set_options, reset_options, rotate_keys"
            )

        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        conn_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        alter_parts = [f"ALTER CONNECTION {conn_table.sql(self.dialect)}"]

        if rotate_keys:
            if set_options or reset_options:
                raise ValueError("Cannot rotate keys and set/reset options together")
            sql = f"{alter_parts[0]} ROTATE KEYS"
        else:
            if set_options:
                options = []
                for key, value in set_options.items():
                    # Check if value is a SECRET() reference
                    if isinstance(value, str) and value.startswith("SECRET("):
                        options.append(f"{key} = {value}")
                    else:
                        # Quote regular values
                        options.append(f"{key} = '{value}'")

                options_str = ", ".join(options)
                alter_parts.append(f"SET ({options_str})")

            if reset_options:
                reset_str = ", ".join(reset_options)
                alter_parts.append(f"RESET ({reset_str})")

            sql = " ".join(alter_parts)

        with self.begin() as cur:
            cur.execute(sql)

    def create_secret(
        self,
        name: str,
        /,
        value: str,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create a secret in Materialize.

        Secrets store sensitive data like passwords, API keys, and certificates.
        They can be referenced in connections and other objects.

        Parameters
        ----------
        name
            Secret name to create.
        value
            Secret value (plain text or base64 encoded).
        database
            Name of the database (catalog) where the secret will be created.
        schema
            Name of the schema where the secret will be created.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.create_secret("kafka_password", "my_secret_password")
        >>> con.create_secret("pg_password", "postgres_pwd")
        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        secret_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        create_sql = f"CREATE SECRET {secret_table.sql(self.dialect)} AS '{value}'"

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_sql)
        con.commit()

    def drop_secret(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a secret.

        Parameters
        ----------
        name
            Secret name to drop.
        database
            Name of the database (catalog) where the secret exists, if not the default.
        schema
            Name of the schema where the secret exists, if not the default.
        force
            If `False`, an exception is raised if the secret does not exist.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_secret("kafka_password", force=True)
        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        secret_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        drop_stmt = sge.Drop(
            this=secret_table,
            kind="SECRET",
            exists=force,
        )

        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def list_secrets(
        self,
        *,
        database: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List secrets in Materialize.

        Parameters
        ----------
        database
            Database/schema to list secrets from.
            If None, uses current database.
        like
            Pattern to filter secret names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of secret names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_secrets()
        ['kafka_password', 'pg_password', 'aws_secret']
        >>> con.list_secrets(like="kafka%")
        ['kafka_password']
        """
        query = """
        SELECT s.name
        FROM mz_catalog.mz_secrets s
        JOIN mz_catalog.mz_schemas sc ON s.schema_id = sc.id
        WHERE sc.name = %(database)s
        """

        params = {"database": database or self.current_database}

        if like is not None:
            query += " AND s.name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY s.name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def alter_secret(
        self,
        name: str,
        /,
        value: str,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Alter a secret's value.

        Updates the value of an existing secret. Future connections, sources, and
        sinks will use the new value immediately. Note that existing running
        sources/sinks may continue caching the old secret for some time.

        Parameters
        ----------
        name
            Secret name to alter.
        value
            New secret value (will be converted to bytea).
        database
            Name of the database (catalog) where the secret exists.
        schema
            Name of the schema where the secret exists.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Update a secret value:

        >>> con.alter_secret("kafka_password", "new_password123")

        Update with base64-encoded value:

        >>> con.alter_secret("ssl_cert", "decode('c2VjcmV0Cg==', 'base64')")

        Notes
        -----
        - Existing sources/sinks may cache the old value for several weeks
        - To force immediate refresh, restart cluster replicas:
          - Managed: ALTER CLUSTER SET (REPLICATION FACTOR = 0) then back to 1
          - Unmanaged: DROP and recreate replicas

        """
        quoted = self.compiler.quoted
        # Note: sqlglot's 'catalog' parameter maps to Materialize's database
        # and sqlglot's 'db' parameter maps to Materialize's schema
        secret_table = sg.table(name, catalog=database, db=schema, quoted=quoted)

        # Check if value contains SQL functions (like decode), don't quote if so
        if "(" in value and ")" in value:
            # Likely a function call like decode('...', 'base64')
            sql = f"ALTER SECRET {secret_table.sql(self.dialect)} AS {value}"
        else:
            # Regular string value, needs quoting
            sql = f"ALTER SECRET {secret_table.sql(self.dialect)} AS '{value}'"

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(sql)
        con.commit()

    def create_cluster(
        self,
        name: str,
        /,
        *,
        size: str | None = None,
        replication_factor: int = 1,
        disk: bool = False,
        introspection_interval: str | None = None,
        introspection_debugging: bool = False,
        managed: bool = True,
    ) -> None:
        """Create a cluster in Materialize.

        Clusters provide resource isolation and computational resources for running
        queries, materialized views, indexes, sources, and sinks.

        Parameters
        ----------
        name
            Cluster name to create.
        size
            Cluster size (e.g., '25cc', '50cc', '100cc', '200cc', etc.).
            Required for managed clusters. Use `list_cluster_sizes()` to discover
            available sizes in your Materialize instance.
        replication_factor
            Number of replicas (default: 1). Set to 0 to create an empty cluster.
        disk
            Whether replicas should have disk storage (default: False).
        introspection_interval
            Introspection data collection interval (default: '1s').
            Set to '0' to disable introspection.
        introspection_debugging
            Enable introspection debugging data (default: False).
        managed
            Whether to create a managed cluster (default: True).
            Unmanaged clusters require manual replica management.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        >>> # Basic cluster
        >>> con.create_cluster("my_cluster", size="100cc")

        >>> # Cluster with high availability
        >>> con.create_cluster("ha_cluster", size="400cc", replication_factor=2)

        >>> # Cluster with disk storage
        >>> con.create_cluster("disk_cluster", size="200cc", disk=True)

        >>> # Cluster with introspection disabled
        >>> con.create_cluster("fast_cluster", size="100cc", introspection_interval="0")

        >>> # Empty cluster (no replicas)
        >>> con.create_cluster("paused_cluster", size="100cc", replication_factor=0)
        """
        if managed and size is None:
            raise ValueError("size is required for managed clusters")

        quoted = self.compiler.quoted
        cluster_id = sg.to_identifier(name, quoted=quoted)

        # Build CREATE CLUSTER statement
        create_parts = [f"CREATE CLUSTER {cluster_id.sql(self.dialect)}"]

        # Build cluster options
        options = []

        if managed:
            if size:
                options.append(f"SIZE = '{size}'")

            if replication_factor != 1:
                options.append(f"REPLICATION FACTOR = {replication_factor}")

            if disk:
                options.append("DISK = TRUE")

            if introspection_interval is not None:
                options.append(f"INTROSPECTION INTERVAL = '{introspection_interval}'")

            if introspection_debugging:
                options.append("INTROSPECTION DEBUGGING = TRUE")
        else:
            options.append("MANAGED = FALSE")

        if options:
            options_str = ", ".join(options)
            create_parts.append(f"({options_str})")

        create_sql = " ".join(create_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(create_sql)
        con.commit()

    def drop_cluster(
        self,
        name: str,
        /,
        *,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        """Drop a cluster.

        Parameters
        ----------
        name
            Cluster name to drop.
        force
            If `False`, an exception is raised if the cluster does not exist.
        cascade
            If `True`, drop dependent objects (indexes, materialized views) as well.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.drop_cluster("my_cluster", force=True)
        >>> con.drop_cluster("old_cluster", cascade=True)
        """
        quoted = self.compiler.quoted
        cluster_id = sg.to_identifier(name, quoted=quoted)

        drop_stmt = sge.Drop(
            this=cluster_id,
            kind="CLUSTER",
            exists=force,
            cascade=cascade,
        )

        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def alter_cluster(
        self,
        name: str,
        /,
        *,
        rename_to: str | None = None,
        set_options: dict[str, Any] | None = None,
        reset_options: list[str] | None = None,
    ) -> None:
        """Alter a cluster's configuration.

        Parameters
        ----------
        name
            Cluster name to alter.
        rename_to
            New name for the cluster (for rename operations).
        set_options
            Dictionary of options to set, e.g.:
            - 'SIZE': Cluster size (e.g., '25cc', '50cc'). Use `list_cluster_sizes()`
              to discover available sizes.
            - 'REPLICATION FACTOR': Number of replicas - does not increase workload capacity (int)
            - 'DISK': Enable disk storage (bool)
            - 'INTROSPECTION INTERVAL': Collection interval (str like '1s', '0' to disable)
            - 'INTROSPECTION DEBUGGING': Enable debugging data (bool)
        reset_options
            List of option names to reset to defaults:
            - 'REPLICATION FACTOR'
            - 'INTROSPECTION INTERVAL'
            - 'INTROSPECTION DEBUGGING'
            - 'SCHEDULE'

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Rename a cluster:

        >>> con.alter_cluster("old_name", rename_to="new_name")

        Resize a cluster:

        >>> con.alter_cluster("my_cluster", set_options={"SIZE": "200cc"})

        Change replication factor:

        >>> con.alter_cluster("my_cluster", set_options={"REPLICATION FACTOR": 2})

        Disable introspection:

        >>> con.alter_cluster("my_cluster", set_options={"INTROSPECTION INTERVAL": "0"})

        Reset options to defaults:

        >>> con.alter_cluster("my_cluster", reset_options=["REPLICATION FACTOR"])

        Notes
        -----
        - Cannot modify the same parameter via both SET and RESET
        - Changes are applied atomically
        - Cluster resizing may take time depending on workload

        """
        if not any([rename_to, set_options, reset_options]):
            raise ValueError(
                "Must specify at least one of: rename_to, set_options, reset_options"
            )

        if rename_to and (set_options or reset_options):
            raise ValueError("Cannot rename and set/reset options in same operation")

        quoted = self.compiler.quoted
        cluster_id = sg.to_identifier(name, quoted=quoted)

        if rename_to:
            # RENAME operation
            new_cluster_id = sg.to_identifier(rename_to, quoted=quoted)
            sql = f"ALTER CLUSTER {cluster_id.sql(self.dialect)} RENAME TO {new_cluster_id.sql(self.dialect)}"
        else:
            # SET/RESET operations
            alter_parts = [f"ALTER CLUSTER {cluster_id.sql(self.dialect)}"]

            if set_options:
                options = []
                for key, value in set_options.items():
                    key_upper = key.upper()
                    if isinstance(value, bool):
                        options.append(f"{key_upper} = {str(value).upper()}")
                    elif isinstance(value, int):
                        options.append(f"{key_upper} = {value}")
                    else:
                        # String values
                        options.append(f"{key_upper} = '{value}'")

                options_str = ", ".join(options)
                alter_parts.append(f"SET ({options_str})")

            if reset_options:
                reset_str = ", ".join(opt.upper() for opt in reset_options)
                alter_parts.append(f"RESET ({reset_str})")

            sql = " ".join(alter_parts)

        con = self.con
        with con.cursor() as cursor:
            cursor.execute(sql)
        con.commit()

    def list_clusters(
        self,
        *,
        like: str | None = None,
    ) -> list[str]:
        """List clusters in Materialize.

        Parameters
        ----------
        like
            Pattern to filter cluster names (SQL LIKE syntax).

        Returns
        -------
        list[str]
            List of cluster names

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> con.list_clusters()
        ['quickstart', 'my_cluster', 'ha_cluster']
        >>> con.list_clusters(like="my%")
        ['my_cluster']
        """
        query = """
        SELECT name
        FROM mz_catalog.mz_clusters
        """

        params = {}

        if like is not None:
            query += " WHERE name LIKE %(like)s"
            params["like"] = like

        query += " ORDER BY name"

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def list_cluster_sizes(self) -> list[str]:
        """List available cluster replica sizes in Materialize.

        Returns
        -------
        list[str]
            List of available cluster size names (e.g., '25cc', '50cc', '100cc')

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> sizes = con.list_cluster_sizes()
        >>> sizes
        ['25cc', '50cc', '100cc', '200cc', '300cc', ...]

        Notes
        -----
        - Available sizes may vary between Materialize deployments
        - The values in this catalog may change
        - Use this method to discover sizes rather than hardcoding them

        """
        query = """
        SELECT size
        FROM mz_catalog.mz_cluster_replica_sizes
        ORDER BY credits_per_hour
        """

        with self.begin() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

    def create_index(
        self,
        name: str,
        /,
        table: str,
        *,
        expressions: list[str] | None = None,
        cluster: str | None = None,
        database: str | None = None,
        unique: bool = False,
    ) -> None:
        """Create an index in Materialize.

        In Materialize, indexes store query results in memory within a specific cluster,
        and keep these results incrementally updated as new data arrives. This ensures that
        indexed data remains fresh, reflecting the latest changes with minimal latency.

        The primary use case for indexes is to accelerate direct queries issued via SELECT statements.
        By maintaining fresh, up-to-date results in memory, indexes can significantly optimize
        query performance, reducing both response time and compute load—especially for resource-intensive
        operations such as joins, aggregations, and repeated subqueries.

        Because indexes are scoped to a single cluster, they are most useful for accelerating
        queries within that cluster. For results that must be shared across clusters or persisted
        to durable storage, consider using a materialized view, which also maintains fresh results
        but is accessible system-wide.

        Parameters
        ----------
        name
            Name of the index to create
        table
            Name of the table, view, or materialized view to index
        expressions
            List of column names or SQL expressions to index. If None, creates
            a default index where Materialize automatically determines the best
            key columns.
        cluster
            Name of the cluster to maintain the index. If None, uses the active
            cluster.
        database
            Schema/database where the index should be created. If None, uses
            the current database.
        unique
            Whether the index enforces uniqueness. This parameter is included
            for API compatibility but is always False for Materialize, as
            Materialize indexes do not enforce uniqueness constraints.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Create a default index (Materialize chooses key columns):

        >>> con.create_index("orders_idx", "orders")

        Create an index on a specific column:

        >>> con.create_index("orders_customer_idx", "orders", expressions=["customer_id"])

        Create a multi-column index:

        >>> con.create_index(
        ...     "orders_composite_idx", "orders", expressions=["customer_id", "order_date"]
        ... )

        Create an index with an expression:

        >>> con.create_index("customers_upper_idx", "customers", expressions=["upper(email)"])

        Create an index in a specific cluster:

        >>> con.create_index("orders_idx", "orders", cluster="production")

        Notes
        -----
        - Default indexes let Materialize automatically choose the best columns
        - Indexes consume memory proportional to the indexed data size
        - Creating indexes on large datasets can take time
        - Materialize indexes only support the 'arrangement' method internally

        """
        if unique:
            # Materialize doesn't support unique indexes
            # Accept the parameter for API compatibility but ignore it
            pass

        quoted = self.compiler.quoted
        idx_name = sg.table(name, quoted=quoted)
        table_name = sg.table(table, quoted=quoted)

        create_parts = [f"CREATE INDEX {idx_name.sql(self.dialect)}"]

        # Add cluster specification if provided
        if cluster is not None:
            cluster_name = sg.table(cluster, quoted=quoted)
            create_parts.append(f"IN CLUSTER {cluster_name.sql(self.dialect)}")

        # Add table reference
        create_parts.append(f"ON {table_name.sql(self.dialect)}")

        # Add expressions if provided
        if expressions is not None:
            expr_list = ", ".join(expressions)
            create_parts.append(f"({expr_list})")

        sql = " ".join(create_parts)

        with self.begin() as cur:
            cur.execute(sql)

    def drop_index(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop an index from Materialize.

        Parameters
        ----------
        name
            Name of the index to drop
        database
            Schema/database where the index exists. If None, uses the current
            database.
        force
            If True, does not raise an error if the index does not exist
            (uses IF EXISTS)

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        Drop an index:

        >>> con.drop_index("orders_idx")

        Drop an index only if it exists:

        >>> con.drop_index("old_index", force=True)

        """
        quoted = self.compiler.quoted
        idx_name = sg.table(name, quoted=quoted)

        drop_cmd = sge.Drop(
            this=idx_name,
            kind="INDEX",
            exists=force,
        )

        sql = drop_cmd.sql(self.dialect)

        with self.begin() as cur:
            cur.execute(sql)

    def list_indexes(
        self,
        *,
        table: str | None = None,
        database: str | None = None,
        cluster: str | None = None,
        like: str | None = None,
    ) -> list[str]:
        """List indexes in Materialize.

        Parameters
        ----------
        table
            Filter indexes for a specific table, view, or materialized view
        database
            Filter indexes by schema/database name
        cluster
            Filter indexes by cluster name
        like
            Filter index names using SQL LIKE pattern (e.g., "orders%")

        Returns
        -------
        list[str]
            List of index names matching the filters

        Examples
        --------
        >>> import ibis
        >>> con = ibis.materialize.connect()

        List all indexes:

        >>> con.list_indexes()
        ['orders_idx', 'customers_idx', ...]

        List indexes on a specific table:

        >>> con.list_indexes(table="orders")
        ['orders_idx', 'orders_customer_idx']

        List indexes in a specific cluster:

        >>> con.list_indexes(cluster="production")
        ['orders_idx', 'products_idx']

        List indexes with a name pattern:

        >>> con.list_indexes(like="orders%")
        ['orders_idx', 'orders_customer_idx', 'orders_composite_idx']

        Combine filters:

        >>> con.list_indexes(table="orders", cluster="production")
        ['orders_idx']

        """
        query_parts = ["SELECT i.name FROM mz_catalog.mz_indexes i"]
        joins = []
        conditions = []
        params = {}

        # Join with schemas if database filter is needed
        if database is not None:
            joins.append("JOIN mz_catalog.mz_schemas s ON i.schema_id = s.id")
            conditions.append("s.name = %(database)s")
            params["database"] = database

        # Join with clusters if cluster filter is needed
        if cluster is not None:
            joins.append("JOIN mz_catalog.mz_clusters c ON i.cluster_id = c.id")
            conditions.append("c.name = %(cluster)s")
            params["cluster"] = cluster

        # Join with relations if table filter is needed
        if table is not None:
            joins.append("JOIN mz_catalog.mz_relations r ON i.on_id = r.id")
            conditions.append("r.name = %(table)s")
            params["table"] = table

        # Add LIKE filter
        if like is not None:
            conditions.append("i.name LIKE %(like)s")
            params["like"] = like

        # Build final query
        if joins:
            query_parts.extend(joins)
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        query_parts.append("ORDER BY i.name")

        query = " ".join(query_parts)

        with self.begin() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]


def connect(
    *,
    user: str | None = None,
    password: str | None = None,
    host: str = "localhost",
    port: int = 6875,  # Materialize default port
    database: str | None = None,
    schema: str | None = None,
    cluster: str | None = None,
    **kwargs: Any,
) -> Backend:
    """Connect to a Materialize database.

    Parameters
    ----------
    user
        Username
    password
        Password
    host
        Hostname (default: localhost)
    port
        Port number (default: 6875 for Materialize)
    database
        Database name
    schema
        Schema name
    cluster
        Default cluster to use for queries. If `None`, uses Materialize's
        default cluster.
    **kwargs
        Additional connection parameters

    Returns
    -------
    Backend
        An Ibis Materialize backend instance

    Examples
    --------
    >>> import ibis
    >>> con = ibis.materialize.connect(
    ...     host="my-materialize.cloud",
    ...     port=6875,
    ...     user="myuser",
    ...     password="mypassword",
    ...     database="materialize",
    ... )

    Connect with a specific cluster:

    >>> con = ibis.materialize.connect(
    ...     host="localhost",
    ...     user="materialize",
    ...     database="materialize",
    ...     cluster="quickstart",
    ... )
    """
    backend = Backend()
    return backend.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        schema=schema,
        cluster=cluster,
        **kwargs,
    )
