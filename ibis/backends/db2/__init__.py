"""IBM DB2 backend for Ibis."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

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

    import pandas as pd


class Backend(SQLBackend):
    """IBM DB2 backend for Ibis."""

    name = "db2"
    supports_temporary_tables = True
    supports_python_udfs = False

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
            Query results
        """
        if isinstance(query, sg.Expression):
            query = query.sql(dialect=self.compiler.dialect)

        with self._safe_raw_sql(query, **kwargs) as cursor:
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
            cursor.execute(query, (table_name.upper(), schema_name))
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
            # Keep column names in uppercase as DB2 stores them
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
        full_name = f"{db_prefix}{name}"

        if overwrite:
            self.drop_table(name, database=database, force=True)

        # Build column definitions
        col_defs = []
        for col_name, col_type in schema.items():
            db2_type = ibis_type_to_db2_type(col_type)
            nullable = "NULL" if col_type.nullable else "NOT NULL"
            col_defs.append(f"{col_name} {db2_type} {nullable}")

        columns_sql = ", ".join(col_defs)
        create_sql = f"CREATE {temp_clause}TABLE {full_name} ({columns_sql})"

        self.raw_sql(create_sql)

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
        full_name = f"{db_prefix}{name}"

        if force:
            # Check if table exists first
            check_sql = f"""
                SELECT COUNT(*)
                FROM SYSCAT.TABLES
                WHERE TABNAME = '{name.upper()}'
                AND TABSCHEMA = COALESCE('{database.upper() if database else ""}', CURRENT SCHEMA)
            """
            with self._safe_raw_sql(check_sql) as cursor:
                exists = cursor.fetchone()[0] > 0

            if not exists:
                return

        drop_sql = f"DROP TABLE {full_name}"
        self.raw_sql(drop_sql)

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
        full_name = f"{db_prefix}{table_name}"

        if overwrite:
            self.raw_sql(f"TRUNCATE TABLE {full_name} IMMEDIATE")

        if isinstance(obj, pd.DataFrame):
            # Batch insert from DataFrame
            if obj.empty:
                return

            columns = ", ".join(obj.columns)
            placeholders = ", ".join(["?" for _ in obj.columns])
            insert_sql = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"

            cursor = self._connection.cursor()
            try:
                # Insert in batches
                batch_size = 1000
                for i in range(0, len(obj), batch_size):
                    batch = obj.iloc[i : i + batch_size]
                    cursor.executemany(insert_sql, batch.values.tolist())
                self._connection.commit()
            finally:
                cursor.close()
        else:
            # Insert from table expression
            insert_sql = f"INSERT INTO {full_name} {self.compile(obj)}"
            self.raw_sql(insert_sql)

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
            # Fetch column names and types
            columns = [desc[0].lower() for desc in cursor.description]

            # Fetch all rows
            rows = cursor.fetchall()

        # Convert to DataFrame
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
) -> Backend:
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
    return backend.connect(
        database=database,
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        schema=schema,
        **kwargs,
    )
