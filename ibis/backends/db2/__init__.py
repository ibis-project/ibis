"""IBM Db2 backend for Ibis."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.db2.converter import Db2PandasData
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.db2 import Db2Compiler

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import pandas as pd


class Backend(SQLBackend):
    """IBM Db2 backend for Ibis."""

    name = "db2"
    compiler = Db2Compiler()
    supports_temporary_tables = True
    supports_python_udfs = False

    def __init__(self, *args, **kwargs):
        """Initialize Db2 backend."""
        super().__init__(*args, **kwargs)
        self._connection = None
        self._cursor = None
        self._conn_str = None  # stored so _reconnect() can open a fresh connection

    @property
    def version(self) -> str:
        """Return the version of the Db2 server."""
        with self._safe_raw_sql(
            "SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO"
        ) as cur:
            result = cur.fetchone()
            return result[0] if result else "unknown"

    def do_connect(
        self,
        database: str,
        hostname: str = "localhost",
        port: int = 50000,
        username: str | None = None,
        password: str | None = None,
        schema: str | None = None,
        ssl: bool = False,
        ssl_server_certificate: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Connect to a Db2 database.

        Parameters
        ----------
        database : str
            Database name
        hostname : str, default "localhost"
            Hostname of Db2 server
        port : int, default 50000
            Port number
        username : str, optional
            Username for authentication
        password : str, optional
            Password for authentication
        schema : str, optional
            Default schema
        ssl : bool, default False
            Enable SSL/TLS encrypted connection. When ``True``, the connection
            string includes ``SECURITY=SSL``.
        ssl_server_certificate : str or Path, optional
            Path to the server's SSL certificate (PEM or ARM format). Passed
            as ``SSLServerCertificate=<path>`` in the connection string. If
            ``ssl`` is ``True`` but this is ``None``, server certificate
            validation is skipped (useful for self-signed certificates in dev
            environments).
        **kwargs
            Additional IBM Db2 connection string key=value parameters.
        """
        import ibm_db
        import ibm_db_dbi

        # Build connection string
        conn_str_parts = [
            f"DATABASE={database}",
            f"HOSTNAME={hostname}",
            f"PORT={port}",
            "PROTOCOL=TCPIP",
        ]

        if username:
            conn_str_parts.append(f"UID={username}")
        if password:
            conn_str_parts.append(f"PWD={password}")

        # SSL parameters — ibm_db uses SECURITY=SSL in the connection string
        if ssl:
            conn_str_parts.append("SECURITY=SSL")
            if ssl_server_certificate is not None:
                conn_str_parts.append(
                    f"SSLServerCertificate={Path(ssl_server_certificate)}"
                )

        # Add any additional connection parameters
        _reserved = {
            "DATABASE",
            "HOSTNAME",
            "PORT",
            "UID",
            "PWD",
            "SECURITY",
            "SSLSERVERCERTIFICATE",
        }
        for key, value in kwargs.items():
            if key.upper() not in _reserved:
                conn_str_parts.append(f"{key.upper()}={value}")

        conn_str = ";".join(conn_str_parts)

        try:
            # Store connection string for later reconnects (e.g. after DDL)
            self._conn_str = conn_str
            # Connect using ibm_db
            ibm_db_conn = ibm_db.connect(conn_str, "", "")
            # Wrap with DBI-compliant interface
            self._connection = ibm_db_dbi.Connection(ibm_db_conn)
            self._cursor = self._connection.cursor()

            # Set schema if provided
            if schema:
                self._cursor.execute(f"SET SCHEMA {schema}")

        except Exception as e:
            raise exc.OperationNotDefinedError(f"Failed to connect to Db2: {e}") from e

    @property
    def con(self):
        """Alias for _connection.

        Required by base SQL backend's _make_memtable_finalizer which
        references self.con.
        """
        return self._connection

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        """Create a Db2 backend from a URL.

        Parameters
        ----------
        url : ParseResult
            Parsed URL object
        **kwarg_overrides
            Additional keyword arguments to override URL parameters

        Returns
        -------
        Self
            Connected Db2 backend instance

        Notes
        -----
        SSL can be enabled via query parameters in the URL::

            db2://user:pass@host:50001/SAMPLE?ssl=true
            db2://user:pass@host:50001/SAMPLE?ssl=true&ssl_server_certificate=/path/to/cert.pem
        """
        kwargs = {}
        database, *schema = url.path[1:].split("/", 1)
        if url.username:
            kwargs["username"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["hostname"] = url.hostname
        if database:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        if schema:
            kwargs["schema"] = schema[0]

        # Parse SSL-related query parameters from the URL
        query_params = (
            dict(pair.split("=", 1) for pair in url.query.split("&") if "=" in pair)
            if url.query
            else {}
        )
        if "ssl" in query_params:
            kwargs["ssl"] = query_params["ssl"].lower() in ("1", "true", "yes")
        if "ssl_server_certificate" in query_params:
            kwargs["ssl_server_certificate"] = query_params["ssl_server_certificate"]

        kwargs.update(kwarg_overrides)
        return self.connect(**kwargs)

    def table(self, name: str, /, *, database: tuple[str, str] | str | None = None):
        """Return a table expression, normalising the name to uppercase for Db2."""
        return super().table(name.upper(), database=database)

    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None

    def _reconnect(self) -> None:
        """Close and immediately reopen the connection using the stored DSN.

        ibm_db_dbi's catalog views (e.g. SYSCAT.COLUMNS) are not always
        visible on the same connection directly after a DDL commit.  A fresh
        connection guarantees up-to-date catalog visibility and also clears
        any internal driver state left over from DDL execution.
        """
        import ibm_db
        import ibm_db_dbi

        self.disconnect()
        ibm_db_conn = ibm_db.connect(self._conn_str, "", "")
        self._connection = ibm_db_dbi.Connection(ibm_db_conn)
        self._cursor = self._connection.cursor()

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str | sg.Expression, **kwargs: Any):
        """Execute raw SQL safely with cursor management."""
        if isinstance(query, sg.exp.Expression):
            query = query.sql(dialect=self.compiler.dialect)

        cursor = self._connection.cursor()
        try:
            cursor.execute(query, **kwargs)
            yield cursor
        finally:
            cursor.close()

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        """Execute a raw SQL query.

        Parameters
        ----------
        query : str | sg.Expression
            SQL query to execute
        **kwargs
            Additional parameters

        Returns
        -------
        Any
            Query results (cursor)
        """
        if isinstance(query, sg.exp.Expression):
            query = query.sql(dialect=self.compiler.dialect)

        # Don't use context manager as it closes the cursor
        cursor = self._connection.cursor()
        try:
            cursor.execute(query, **kwargs)
        except Exception:
            cursor.close()
            raise
        else:
            return cursor

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame.from_records(
            cursor.fetchall(), columns=schema.names, coerce_float=True
        )
        return Db2PandasData.convert_table(df, schema)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List tables in the database.

        Parameters
        ----------
        like : str, optional
            Pattern to filter table names
        database : str, optional
            Database/schema name

        Returns
        -------
        list[str]
            List of table names
        """
        query = """
            SELECT TABNAME
            FROM SYSCAT.TABLES
            WHERE TABSCHEMA = CURRENT SCHEMA
            AND TYPE = 'T'
        """

        if like:
            query += f" AND UPPER(TABNAME) LIKE UPPER('{like}')"

        query += " ORDER BY TABNAME"

        with self._safe_raw_sql(query) as cursor:
            # Db2 stores names in uppercase; return lowercase for ibis convention
            return [row[0].lower() for row in cursor.fetchall()]

    def list_databases(self, like: str | None = None) -> list[str]:
        """List schemas in the database.

        Parameters
        ----------
        like : str, optional
            Pattern to filter schema names

        Returns
        -------
        list[str]
            List of schema names
        """
        query = """
            SELECT SCHEMANAME
            FROM SYSCAT.SCHEMATA
            WHERE SCHEMANAME NOT LIKE 'SYS%'
        """

        if like:
            query += f" AND SCHEMANAME LIKE '{like}'"

        query += " ORDER BY SCHEMANAME"

        with self._safe_raw_sql(query) as cursor:
            return [row[0] for row in cursor.fetchall()]

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Get the schema of a table.

        Parameters
        ----------
        table_name : str
            Name of the table
        catalog : str, optional
            Catalog name (unused in Db2)
        database : str, optional
            Schema name

        Returns
        -------
        sch.Schema
            Table schema
        """
        query = """
            SELECT COLNAME, TYPENAME, LENGTH, SCALE, NULLS
            FROM SYSCAT.COLUMNS
            WHERE TABNAME = ?
            AND TABSCHEMA = COALESCE(?, CURRENT SCHEMA)
            ORDER BY COLNO
        """

        schema_name = database or self.current_database

        cursor = self._connection.cursor()
        try:
            # SYSCAT.COLUMNS stores unquoted names in UPPERCASE and quoted names
            # verbatim.  Try the name as-given first (handles both quoted-lowercase
            # tables created by create_table and any other exact-case names), then
            # fall back to UPPERCASE (handles tables from unquoted DDL / data load).
            cursor.execute(query, (table_name, schema_name))
            rows = cursor.fetchall()
            if not rows:
                cursor.execute(query, (table_name.upper(), schema_name))
                rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            raise exc.IbisError(f"Table not found: {table_name}")

        type_mapper = self.compiler.type_mapper
        fields = {}
        for col_name, type_name, length, scale, nulls in rows:
            # Reconstruct the full type string so SQLGlot's Db2 dialect parser
            # can handle it — this delegates all type parsing to the same
            # type_mapper (Db2Type / SqlglotType.from_string) that the rest of
            # the compiler uses, rather than duplicating the mapping by hand.
            if type_name in ("DECIMAL", "NUMERIC"):
                type_str = f"{type_name}({length},{scale})"
            elif type_name in ("VARCHAR", "CHAR", "VARBINARY"):
                type_str = f"{type_name}({length})"
            else:
                type_str = type_name

            ibis_type = type_mapper.from_string(type_str, nullable=(nulls == "Y"))
            # Return column names verbatim — SYSCAT stores them in the case they
            # were defined with (uppercase for unquoted DDL, exact case for quoted).
            fields[col_name] = ibis_type

        return sch.Schema(fields)

    @property
    def current_database(self) -> str:
        """Return the current schema."""
        with self._safe_raw_sql("VALUES CURRENT SCHEMA") as cursor:
            result = cursor.fetchone()
            return result[0] if result else None

    def create_table(
        self,
        name: str,
        obj: ir.Table | pd.DataFrame | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table.

        Parameters
        ----------
        name : str
            Table name
        obj : ir.Table | pd.DataFrame, optional
            Data to insert
        schema : sch.Schema, optional
            Table schema
        database : str, optional
            Schema name
        temp : bool, default False
            Create temporary table
        overwrite : bool, default False
            Overwrite if exists

        Returns
        -------
        ir.Table
            Table expression
        """
        import pandas as pd

        if obj is None and schema is None:
            raise exc.IbisError("Either obj or schema must be provided")

        # Db2 stores unquoted identifiers in UPPERCASE in SYSCAT.  Normalise
        # the name to uppercase now so the quoted DDL ("NAME") also stores an
        # uppercase name and self.table() / get_schema() can always find it.
        name = name.upper()

        # Normalise schema: accept dict.items() (tuples) and plain dicts in
        # addition to ibis.Schema objects.
        if schema is not None and not isinstance(schema, sch.Schema):
            schema = sch.schema(schema)

        # Normalise obj: convert pyarrow.Table to pandas DataFrame so the rest
        # of the method can treat obj as either pd.DataFrame or ir.Table.
        try:
            import pyarrow as pa

            if isinstance(obj, pa.RecordBatchReader):
                obj = obj.read_all().to_pandas()
            elif isinstance(obj, (pa.Table, pa.RecordBatch)):
                obj = obj.to_pandas()
        except ImportError:
            pass

        if schema is None:
            if isinstance(obj, pd.DataFrame):
                schema = sch.infer(obj)
            elif isinstance(obj, ir.Table):
                schema = obj.schema()
            else:
                obj = pd.DataFrame(obj)
                schema = sch.infer(obj)

        # Build CREATE TABLE statement
        temp_clause = "GLOBAL TEMPORARY " if temp else ""
        # sg.table(..., quoted=True) builds the table reference the same way
        # Ibis/SQLGlot quotes it in SELECT, so CREATE stays consistent by
        # construction rather than by hand-replicating the quoting rules.
        full_name = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            self.dialect
        )

        if overwrite:
            self.drop_table(name, database=database, force=True)

        # Build column definitions — delegate type rendering to the compiler's
        # type_mapper (Db2Type) so that the same SQLGlot Db2 dialect that drives
        # SELECT generation also drives CREATE TABLE column types.
        import ibis.expr.datatypes as dt

        col_defs = []
        type_mapper = self.compiler.type_mapper
        for col_name, col_type in schema.items():
            # Accept string type names (e.g. "int32") in addition to DataType objects
            if isinstance(col_type, str):
                col_type = dt.dtype(col_type)
            db2_type = type_mapper.to_string(col_type)
            # DB2 does not accept bare NULL as a column constraint — nullable is
            # the default, so only emit NOT NULL for non-nullable columns.
            null_constraint = "" if col_type.nullable else " NOT NULL"
            # sg.to_identifier(..., quoted=True) is the same quoting primitive
            # SQLGlot uses for column references, so column names always match
            quoted_col_name = sg.to_identifier(
                col_name, quoted=self.compiler.quoted
            ).sql(self.dialect)
            col_defs.append(f"{quoted_col_name} {db2_type}{null_constraint}")

        columns_sql = ", ".join(col_defs)
        # DB2 requires an explicit tablespace with sufficient page size.
        # IBIS_32K (32KB pages) is pre-created in conftest._load_data and
        # supports all column types including VARCHAR(32768) used by
        # memtable/cache operations.
        tablespace_clause = " IN IBIS_32K" if not temp else ""
        create_sql = (
            f"CREATE {temp_clause}TABLE {full_name} ({columns_sql}){tablespace_clause}"
        )

        with self._safe_raw_sql(create_sql):
            pass
        # Commit the CREATE TABLE statement so it is visible to new connections.
        self._connection.commit()

        # Only reconnect for permanent tables — GLOBAL TEMPORARY tables are
        # session-scoped and get destroyed if the connection is recycled.
        if not temp:
            self._reconnect()

        # Insert data if provided
        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                self.insert(name, obj, database=database)
            elif isinstance(obj, ir.Table):
                # Ensure any in-memory tables referenced by obj (e.g. memtables)
                # are registered in the database before the INSERT is executed.
                self._register_in_memory_tables(obj)
                insert_sql = f"INSERT INTO {full_name} {self.compile(obj)}"
                with self._safe_raw_sql(insert_sql):
                    pass
            else:
                # Fallback: list of dicts etc.
                self.insert(name, pd.DataFrame(obj), database=database)

        return self.table(name, database=database)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name : str
            Table name
        database : str, optional
            Schema name
        force : bool, default False
            Suppress errors if table doesn't exist
        """
        name = name.upper()
        full_name = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            self.dialect
        )

        if force:
            # Check if table exists first using parameterized query
            cursor = self._connection.cursor()
            try:
                if database:
                    check_sql = """
                        SELECT COUNT(*)
                        FROM SYSCAT.TABLES
                        WHERE TABNAME = ?
                        AND TABSCHEMA = ?
                    """
                    cursor.execute(check_sql, (name, database.upper()))
                else:
                    check_sql = """
                        SELECT COUNT(*)
                        FROM SYSCAT.TABLES
                        WHERE TABNAME = ?
                        AND TABSCHEMA = CURRENT SCHEMA
                    """
                    cursor.execute(check_sql, (name,))

                exists = cursor.fetchone()[0] > 0
            finally:
                cursor.close()

            if not exists:
                return

        drop_sql = f"DROP TABLE {full_name}"
        with self._safe_raw_sql(drop_sql):
            pass
        # Commit the DROP TABLE statement
        self._connection.commit()

    def insert(
        self,
        name: str,
        /,
        obj: pd.DataFrame | ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        Parameters
        ----------
        name : str
            Target table name
        obj : pd.DataFrame | ir.Table
            Data to insert
        database : str, optional
            Schema name
        overwrite : bool, default False
            Truncate table before insert
        """
        import pandas as pd

        name = name.upper()
        full_name = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            self.dialect
        )

        if overwrite:
            # Commit any open transaction first to ensure TRUNCATE can be first statement
            self._connection.commit()
            # TRUNCATE TABLE ... IMMEDIATE must be first statement in transaction
            with self._safe_raw_sql(f"TRUNCATE TABLE {full_name} IMMEDIATE"):
                pass
            # Commit the TRUNCATE to complete the transaction
            self._connection.commit()

        if isinstance(obj, pd.DataFrame):
            # Batch insert from DataFrame
            if obj.empty:
                return

            # Same quoting primitive SQLGlot uses for column references
            quoted_columns = [
                sg.to_identifier(col, quoted=self.compiler.quoted).sql(self.dialect)
                for col in obj.columns
            ]
            columns = ", ".join(quoted_columns)
            placeholders = ", ".join(["?" for _ in obj.columns])
            insert_sql = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"  # noqa: S608

            cursor = self._connection.cursor()
            try:
                # Insert in batches
                batch_size = 1000
                for i in range(0, len(obj), batch_size):
                    batch = obj.iloc[i : i + batch_size]
                    # Convert NaN/NaT/pd.NA to None for Db2 compatibility
                    rows = self._convert_dataframe_to_rows(batch)
                    cursor.executemany(insert_sql, rows)
                self._connection.commit()
            finally:
                cursor.close()
        else:
            # Insert from table expression — ensure any in-memory tables
            # referenced (e.g. memtables) are registered first.
            self._register_in_memory_tables(obj)
            insert_sql = f"INSERT INTO {full_name} {self.compile(obj)}"
            with self._safe_raw_sql(insert_sql):
                pass

    @staticmethod
    def _convert_dataframe_to_rows(df: pd.DataFrame) -> list[tuple]:
        """Convert DataFrame to list of tuples, replacing NaN/NaT/pd.NA with None.

        This is necessary because Db2's ibm_db_dbi driver expects SQL NULL values
        to be represented as Python None.
        """
        import pandas as pd

        df = df.astype(object).where(pd.notnull(df), None)
        return list(df.itertuples(index=False, name=None))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get schema from a SQL query.

        Parameters
        ----------
        query : str
            SQL query

        Returns
        -------
        sch.Schema
            Query result schema
        """
        with self._safe_raw_sql(query) as cursor:
            if not cursor.description:
                return sch.Schema({})

            fields = {}
            for col_desc in cursor.description:
                col_name = col_desc[0].lower()
                # Use a simple string type for now, can be enhanced later
                from ibis.backends.db2.datatypes import parse_db2_type

                fields[col_name] = parse_db2_type("VARCHAR")

            return sch.Schema(fields)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """Register an in-memory table.

        Parameters
        ----------
        op : ops.InMemoryTable
            In-memory table operation
        """
        import pandas as pd

        # Create a temporary table from the in-memory data
        name = op.name
        data = op.data.to_frame()

        if isinstance(data, pd.DataFrame):
            # Use op.schema (the declared schema) instead of inferring from the
            # DataFrame.  For empty tables, pandas assigns object dtype to every
            # column; sch.infer then maps that to dt.Null, which Db2Type renders
            # as the SQL keyword NULL — producing invalid DDL like
            # "CREATE TABLE t (x NULL NULL)" and SQL0204N "NULL" is an undefined
            # name.  The declared schema always carries the correct column types.
            schema = op.schema
            # Use temp=False — DB2 GLOBAL TEMPORARY tables are session-scoped
            # and get destroyed by _reconnect() called inside create_table.
            # We use a permanent table instead and rely on the finalizer to drop it.
            if name not in self.list_tables():
                self.create_table(
                    name, data, schema=schema, temp=False, overwrite=False
                )


def connect(
    database: str,
    hostname: str = "localhost",
    port: int = 50000,
    username: str | None = None,
    password: str | None = None,
    schema: str | None = None,
    ssl: bool = False,
    ssl_server_certificate: str | Path | None = None,
    **kwargs,
):
    """Connect to a Db2 database.

    Parameters
    ----------
    database : str
        Database name to connect to
    hostname : str, default "localhost"
        Hostname of the Db2 server
    port : int, default 50000
        Port number of the Db2 server
    username : str, optional
        Username for authentication
    password : str, optional
        Password for authentication
    schema : str, optional
        Default schema to use
    ssl : bool, default False
        Enable SSL/TLS encrypted connection. When ``True``, ``SECURITY=SSL``
        is added to the ibm_db connection string. The default Db2 SSL port
        is 50001.
    ssl_server_certificate : str or Path, optional
        Path to the server's SSL certificate (PEM or ARM format). Maps to
        ``SSLServerCertificate=<path>`` in the connection string. When
        ``ssl=True`` and this is ``None``, server certificate validation is
        skipped (useful for self-signed certificates in dev environments).
    **kwargs
        Additional IBM Db2 connection string key=value parameters.

    Returns
    -------
    Backend
        An Ibis Db2 backend instance

    Examples
    --------
    >>> import ibis
    >>> con = ibis.db2.connect(
    ...     database="SAMPLE",
    ...     hostname="localhost",
    ...     port=50000,
    ...     username="db2inst1",
    ...     password="password",
    ... )  # doctest: +SKIP
    >>> con.list_tables()  # doctest: +SKIP
    ['EMPLOYEE', 'DEPARTMENT', 'PROJECT']

    Connect with SSL (no certificate validation):

    >>> con = ibis.db2.connect(
    ...     database="SAMPLE",
    ...     hostname="localhost",
    ...     port=50001,
    ...     username="db2inst1",
    ...     password="password",
    ...     ssl=True,
    ... )  # doctest: +SKIP

    Connect with SSL and a server certificate:

    >>> con = ibis.db2.connect(
    ...     database="SAMPLE",
    ...     hostname="db2-server.example.com",
    ...     port=50001,
    ...     username="db2inst1",
    ...     password="password",
    ...     ssl=True,
    ...     ssl_server_certificate="/path/to/server.arm",
    ... )  # doctest: +SKIP
    """
    backend = Backend()
    backend.do_connect(
        database=database,
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        schema=schema,
        ssl=ssl,
        ssl_server_certificate=ssl_server_certificate,
        **kwargs,
    )
    return backend
