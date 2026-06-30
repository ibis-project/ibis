"""IBM DB2 backend for Ibis."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.sql import SQLBackend
from ibis.backends.db2.datatypes import DB2PandasData, parse_db2_type, ibis_type_to_db2_type

if TYPE_CHECKING:
    from collections.abc import Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    from typing_extensions import Self


class Backend(SQLBackend):
    """IBM DB2 backend for Ibis."""

    name = "db2"
    supports_temporary_tables = True
    supports_python_udfs = False

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Always quote an identifier (table or column name) to match
        Ibis/SQLGlot's behavior when reading.

        Ibis always quotes identifiers when generating SELECT statements.
        We must quote consistently in CREATE/INSERT/DROP to avoid case
        mismatches. This single method is used for both table names and
        column names, since DB2 quoting rules are identical for both.

        Parameters
        ----------
        name : str
            Identifier name (table or column name)

        Returns
        -------
        str
            Quoted identifier
        """
        # Always quote - no exceptions
        # Escape any existing double quotes by doubling them
        escaped_name = name.replace('"', '""')
        return f'"{escaped_name}"'

    @property
    def compiler(self):
        """Lazy load the compiler to avoid circular imports."""
        from ibis.backends.sql.compilers.db2 import DB2Compiler
        if not hasattr(self, '_compiler'):
            self._compiler = DB2Compiler()
        return self._compiler

    def __init__(self, *args, **kwargs):
        """Initialize DB2 backend."""
        super().__init__(*args, **kwargs)
        self._connection = None
        self._cursor = None

    @property
    def version(self) -> str:
        """Return the version of the DB2 server."""
        with self._safe_raw_sql("SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO") as cur:
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
        **kwargs: Any,
    ) -> None:
        """
        Connect to a DB2 database.

        Parameters
        ----------
        database : str
            Database name
        hostname : str, default "localhost"
            Hostname of DB2 server
        port : int, default 50000
            Port number
        username : str, optional
            Username for authentication
        password : str, optional
            Password for authentication
        schema : str, optional
            Default schema
        **kwargs
            Additional connection parameters
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

        # Add any additional connection parameters
        for key, value in kwargs.items():
            if key.upper() not in ("DATABASE", "HOSTNAME", "PORT", "UID", "PWD"):
                conn_str_parts.append(f"{key.upper()}={value}")

        conn_str = ";".join(conn_str_parts)

        try:
            # Connect using ibm_db
            ibm_db_conn = ibm_db.connect(conn_str, "", "")
            # Wrap with DBI-compliant interface
            self._connection = ibm_db_dbi.Connection(ibm_db_conn)
            self._cursor = self._connection.cursor()

            # Set schema if provided
            if schema:
                self._cursor.execute(f"SET SCHEMA {schema}")

        except Exception as e:
            raise exc.OperationNotDefinedError(
                f"Failed to connect to DB2: {e}"
            ) from e

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        """Create a DB2 backend from a URL.
        
        Parameters
        ----------
        url : ParseResult
            Parsed URL object
        **kwarg_overrides
            Additional keyword arguments to override URL parameters
            
        Returns
        -------
        Self
            Connected DB2 backend instance
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
        kwargs.update(kwarg_overrides)
        return self.connect(**kwargs)

    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str, **kwargs: Any):
        """Execute raw SQL safely with cursor management."""
        cursor = self._connection.cursor()
        try:
            cursor.execute(query, **kwargs)
            yield cursor
        finally:
            cursor.close()

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        """
        Execute a raw SQL query.

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

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """
        List tables in the database.

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
            query += f" AND TABNAME LIKE '{like}'"

        query += " ORDER BY TABNAME"

        with self._safe_raw_sql(query) as cursor:
            return [row[0] for row in cursor.fetchall()]

    def list_databases(self, like: str | None = None) -> list[str]:
        """
        List schemas in the database.

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
        """
        Get the schema of a table.

        Parameters
        ----------
        table_name : str
            Name of the table
        catalog : str, optional
            Catalog name (unused in DB2)
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
            # Use exact table name - no uppercasing since we always quote in CREATE
            cursor.execute(query, (table_name, schema_name))
            rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            raise exc.IbisError(f"Table not found: {table_name}")

        fields = {}
        for col_name, type_name, length, scale, nulls in rows:
            # Build type string with parameters
            if type_name in ("DECIMAL", "NUMERIC"):
                type_str = f"{type_name}({length},{scale})"
            elif type_name in ("VARCHAR", "CHAR", "VARBINARY"):
                type_str = f"{type_name}({length})"
            else:
                type_str = type_name

            ibis_type = parse_db2_type(type_str)
            # Set nullable based on NULLS column
            ibis_type = ibis_type(nullable=(nulls == "Y"))
            # Column names are stored in exact case as created (quoted)
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
        """
        Create a new table.

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

        if schema is None:
            if isinstance(obj, pd.DataFrame):
                schema = sch.infer(obj)
            else:
                schema = obj.schema()

        # Build CREATE TABLE statement
        temp_clause = "GLOBAL TEMPORARY " if temp else ""
        db_prefix = f"{database}." if database else ""
        # Always quote table name for consistency with Ibis/SQLGlot
        quoted_name = self._quote_identifier(name)
        full_name = f"{db_prefix}{quoted_name}"

        if overwrite:
            self.drop_table(name, database=database, force=True)

        # Build column definitions
        col_defs = []
        for col_name, col_type in schema.items():
            db2_type = ibis_type_to_db2_type(col_type)
            nullable = "NULL" if col_type.nullable else "NOT NULL"
            # Always quote column name for consistency with Ibis/SQLGlot
            quoted_col_name = self._quote_identifier(col_name)
            col_defs.append(f"{quoted_col_name} {db2_type} {nullable}")

        columns_sql = ", ".join(col_defs)
        create_sql = f"CREATE {temp_clause}TABLE {full_name} ({columns_sql})"

        self.raw_sql(create_sql)
        # Commit the CREATE TABLE statement
        self._connection.commit()

        # Insert data if provided
        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                self.insert(name, obj, database=database)
            else:
                # Insert from table expression
                insert_sql = f"INSERT INTO {full_name} {self.compile(obj)}"
                self.raw_sql(insert_sql)

        return self.table(name, database=database)

    def drop_table(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """
        Drop a table.

        Parameters
        ----------
        name : str
            Table name
        database : str, optional
            Schema name
        force : bool, default False
            Suppress errors if table doesn't exist
        """
        db_prefix = f"{database}." if database else ""
        # Always quote table name for consistency with Ibis/SQLGlot
        quoted_name = self._quote_identifier(name)
        full_name = f"{db_prefix}{quoted_name}"

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
                    # Use exact name - no uppercasing since we always quote in CREATE
                    cursor.execute(check_sql, (name, database.upper()))
                else:
                    check_sql = """
                        SELECT COUNT(*)
                        FROM SYSCAT.TABLES
                        WHERE TABNAME = ?
                        AND TABSCHEMA = CURRENT SCHEMA
                    """
                    # Use exact name - no uppercasing since we always quote in CREATE
                    cursor.execute(check_sql, (name,))
                
                exists = cursor.fetchone()[0] > 0
            finally:
                cursor.close()

            if not exists:
                return

        drop_sql = f"DROP TABLE {full_name}"
        self.raw_sql(drop_sql)
        # Commit the DROP TABLE statement
        self._connection.commit()

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Insert data into a table.

        Parameters
        ----------
        table_name : str
            Target table name
        obj : pd.DataFrame | ir.Table
            Data to insert
        database : str, optional
            Schema name
        overwrite : bool, default False
            Truncate table before insert
        """
        import pandas as pd

        db_prefix = f"{database}." if database else ""
        # Always quote table name for consistency with Ibis/SQLGlot
        quoted_table_name = self._quote_identifier(table_name)
        full_name = f"{db_prefix}{quoted_table_name}"

        if overwrite:
            # Commit any open transaction first to ensure TRUNCATE can be first statement
            self._connection.commit()
            # TRUNCATE TABLE ... IMMEDIATE must be first statement in transaction
            self.raw_sql(f"TRUNCATE TABLE {full_name} IMMEDIATE")
            # Commit the TRUNCATE to complete the transaction
            self._connection.commit()

        if isinstance(obj, pd.DataFrame):
            # Batch insert from DataFrame
            if obj.empty:
                return

            # Always quote column names for consistency with Ibis/SQLGlot
            quoted_columns = [self._quote_identifier(col) for col in obj.columns]
            columns = ", ".join(quoted_columns)
            placeholders = ", ".join(["?" for _ in obj.columns])
            insert_sql = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"

            cursor = self._connection.cursor()
            try:
                # Insert in batches
                batch_size = 1000
                for i in range(0, len(obj), batch_size):
                    batch = obj.iloc[i : i + batch_size]
                    # Convert NaN/NaT/pd.NA to None for DB2 compatibility
                    rows = self._convert_dataframe_to_rows(batch)
                    cursor.executemany(insert_sql, rows)
                self._connection.commit()
            finally:
                cursor.close()
        else:
            # Insert from table expression
            insert_sql = f"INSERT INTO {full_name} {self.compile(obj)}"
            self.raw_sql(insert_sql)
    
    @staticmethod
    def _convert_dataframe_to_rows(df: pd.DataFrame) -> list[tuple]:
        """Convert DataFrame to list of tuples, converting NaN/NaT/pd.NA to None.
        
        This is necessary because DB2's ibm_db_dbi driver's executemany() infers
        parameter types from the first row. If the first row has a real value and
        a later row has NaN, the types are inconsistent, causing SQL0302N errors.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert
            
        Returns
        -------
        list[tuple]
            List of tuples with NaN/NaT/pd.NA converted to None
        """
        import math
        import pandas as pd
        
        def _convert_row(row):
            """Convert a single row, replacing NaN/NaT/pd.NA with None."""
            result = []
            for val in row:
                # Check for None first
                if val is None:
                    result.append(None)
                # Check for float NaN
                elif isinstance(val, float) and math.isnan(val):
                    result.append(None)
                # Check for pandas NA types (NaT, pd.NA, etc.)
                else:
                    try:
                        if pd.isna(val):
                            result.append(None)
                            continue
                    except (TypeError, ValueError):
                        # Not a pandas NA type, keep original value
                        pass
                    result.append(val)
            return tuple(result)
        
        # Use itertuples() instead of values.tolist() to preserve types
        # values.tolist() can upcast integer columns to float when NaN is present
        rows = [_convert_row(row) for row in df.itertuples(index=False, name=None)]
        return rows

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ):
        """
        Execute expression and return results as PyArrow table.

        Parameters
        ----------
        expr : ir.Expr
            Ibis expression
        params : Mapping[ir.Scalar, Any], optional
            Query parameters
        limit : int | str, optional
            Result limit
        **kwargs
            Additional arguments

        Returns
        -------
        pyarrow.Table
            Query results
        """
        import pyarrow as pa

        df = self.to_pandas(expr, params=params, limit=limit, **kwargs)
        return pa.Table.from_pandas(df)

    def to_pandas(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Execute expression and return results as pandas DataFrame.

        Parameters
        ----------
        expr : ir.Expr
            Ibis expression
        params : Mapping[ir.Scalar, Any], optional
            Query parameters
        limit : int | str, optional
            Result limit
        **kwargs
            Additional arguments

        Returns
        -------
        pd.DataFrame
            Query results
        """
        import pandas as pd

        sql = self.compile(expr, params=params, limit=limit)

        with self._safe_raw_sql(sql) as cursor:
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Get column names from the expression's schema to preserve exact case
            # This ensures DataFrame columns match what schema() reports
            schema = expr.as_table().schema()
            columns = list(schema.names)
            
            # Verify column count matches (safety check for alignment)
            if cursor.description and len(cursor.description) != len(columns):
                raise exc.IbisError(
                    f"Column count mismatch: query returned {len(cursor.description)} columns "
                    f"but schema has {len(columns)} columns"
                )

        # Convert to DataFrame with exact column names from schema
        df = pd.DataFrame(rows, columns=columns)

        # Pandas handles most type conversions automatically
        # Additional type conversions can be added here if needed

        return df

    def execute(self, expr: ir.Expr, **kwargs: Any) -> Any:
        """
        Execute an Ibis expression.

        Parameters
        ----------
        expr : ir.Expr
            Expression to execute
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Execution result
        """
        return self.to_pandas(expr, **kwargs)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """
        Get schema from a SQL query.

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
                fields[col_name] = parse_db2_type("VARCHAR")

            return sch.Schema(fields)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """
        Register an in-memory table.

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
            schema = sch.infer(data)
            self.create_table(name, data, schema=schema, temp=True)


def connect(
    database: str,
    hostname: str = "localhost",
    port: int = 50000,
    username: str | None = None,
    password: str | None = None,
    schema: str | None = None,
    **kwargs,
):
    """
    Connect to a DB2 database.

    Parameters
    ----------
    database : str
        Database name to connect to
    hostname : str, default "localhost"
        Hostname of the DB2 server
    port : int, default 50000
        Port number of the DB2 server
    username : str, optional
        Username for authentication
    password : str, optional
        Password for authentication
    schema : str, optional
        Default schema to use
    **kwargs
        Additional connection parameters

    Returns
    -------
    Backend
        An Ibis DB2 backend instance

    Examples
    --------
    >>> import ibis
    >>> con = ibis.db2.connect(
    ...     database="SAMPLE",
    ...     hostname="localhost",
    ...     port=50000,
    ...     username="db2inst1",
    ...     password="password"
    ... )  # doctest: +SKIP
    >>> con.list_tables()  # doctest: +SKIP
    ['EMPLOYEE', 'DEPARTMENT', 'PROJECT']
    """
    backend = Backend()
    backend.do_connect(
        database=database,
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        schema=schema,
        **kwargs,
    )
    return backend