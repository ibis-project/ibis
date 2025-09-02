"""The SingleStoreDB backend."""

# ruff: noqa: BLE001, S110, S608, SIM105 - Performance optimization methods require comprehensive exception handling

from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.singlestoredb import compiler

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from singlestoredb.connection import Connection


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "singlestoredb"
    supports_create_or_replace = True
    supports_temporary_tables = True
    compiler = compiler

    def _fetch_from_cursor(self, cursor, schema):
        """Fetch data from cursor using SingleStoreDB-specific data converter."""
        import pandas as pd

        from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            cursor.close()
            raise

        return SingleStoreDBPandasData.convert_table(df, schema)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr,
        /,
        *,
        params=None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ):
        """Convert expression to PyArrow record batches.

        This method ensures proper data type conversion, particularly for
        boolean values that come from TINYINT(1) columns and JSON columns.
        """
        import json

        import pyarrow as pa

        self._run_pre_execute_hooks(expr)

        # Get the expected schema and compile the query
        schema = expr.as_table().schema()
        sql = self.compile(expr, limit=limit, params=params)

        # Fetch data using our converter
        with self.begin() as cursor:
            cursor.execute(sql)
            df = self._fetch_from_cursor(cursor, schema)

        # Handle JSON columns for PyArrow compatibility
        # PyArrow expects JSON data as strings, but our converter returns parsed objects
        import ibis.expr.datatypes as dt

        for col_name, col_type in schema.items():
            if isinstance(col_type, dt.JSON) and col_name in df.columns:
                # Convert JSON objects back to JSON strings for PyArrow
                def json_to_string(val):
                    if val is None:
                        # For JSON columns, None should become 'null' JSON string
                        # But we need to distinguish between JSON null and SQL NULL
                        # JSON null should be 'null', SQL NULL should remain None
                        # Since our converter already parsed JSON, None here means JSON null
                        return "null"
                    elif isinstance(val, str):
                        # Already a string, ensure it's valid JSON
                        try:
                            # Parse and re-serialize to ensure consistent formatting
                            return json.dumps(json.loads(val))
                        except (json.JSONDecodeError, ValueError):
                            # Not valid JSON, return as string
                            return json.dumps(val)
                    else:
                        # Convert Python object to JSON string
                        return json.dumps(val)

                df[col_name] = df[col_name].map(json_to_string)

        # Convert to PyArrow table with proper type conversion
        table = pa.Table.from_pandas(
            df, schema=schema.to_pyarrow(), preserve_index=False
        )
        return table.to_reader(max_chunksize=chunk_size)

    @property
    def con(self):
        """Return the database connection for compatibility with base class."""
        return self._client

    @util.experimental
    @classmethod
    def from_connection(cls, con: Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to a MySQL database.

        Parameters
        ----------
        con
            An existing connection to a MySQL database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend._client = con
        new_backend._post_connect()
        return new_backend

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @property
    def show(self) -> Any:
        """Access to SHOW commands on the server."""
        return self.con.show

    @property
    def globals(self) -> Any:
        """Accessor for global variables in the server."""
        return self.con.globals

    @property
    def locals(self) -> Any:
        """Accessor for local variables in the server."""
        return self.con.locals

    @property
    def cluster_globals(self) -> Any:
        """Accessor for cluster global variables in the server."""
        return self.con.cluster_globals

    @property
    def cluster_locals(self) -> Any:
        """Accessor for cluster local variables in the server."""
        return self.con.cluster_locals

    @property
    def vars(self) -> Any:
        """Accessor for variables in the server."""
        return self.con.vars

    @property
    def cluster_vars(self) -> Any:
        """Accessor for cluster variables in the server."""
        return self.con.cluster_vars

    @property
    def current_database(self) -> str:
        """Return the current database name."""
        with self.begin() as cur:
            cur.execute("SELECT DATABASE()")
            (database,) = cur.fetchone()
        return database

    @property
    def database(self) -> str:
        """Return the current database name (alias for current_database)."""
        return self.current_database

    @property
    def dialect(self) -> str:
        """Return the SQLGlot dialect name."""
        return "singlestore"

    @classmethod
    def _from_url(cls, url: ParseResult, **kwargs) -> Backend:
        """Create a SingleStoreDB backend from a connection URL."""
        database = url.path[1:] if url.path and len(url.path) > 1 else None

        # Parse query parameters from URL
        query_params = dict(parse_qsl(url.query))

        # Merge query parameters with explicit kwargs, with explicit kwargs taking precedence
        merged_kwargs = {**query_params, **kwargs}

        backend = cls()
        backend.do_connect(
            host=url.hostname or None,
            port=url.port or None,
            user=url.username or None,
            password=unquote_plus(url.password) if url.password is not None else None,
            database=database or None,
            driver=url.scheme or None,
            **merged_kwargs,
        )
        return backend

    def create_database(self, name: str, force: bool = False) -> None:
        """Create a database in SingleStore.

        Parameters
        ----------
        name
            Name of the database to create
        force
            If True, use CREATE DATABASE IF NOT EXISTS

        Examples
        --------
        >>> con.create_database("my_database")
        >>> con.create_database("my_database", force=True)  # Won't fail if exists
        """
        sql = sge.Create(
            kind="DATABASE", exists=force, this=sg.to_identifier(name)
        ).sql(self.dialect)
        with self.begin() as cur:
            cur.execute(sql)

    def drop_database(
        self, name: str, *, catalog: str | None = None, force: bool = False
    ) -> None:
        """Drop a database from SingleStore.

        Parameters
        ----------
        name
            Name of the database to drop
        catalog
            Name of the catalog (not used in SingleStore, for compatibility)
        force
            If True, use DROP DATABASE IF EXISTS to avoid errors if database doesn't exist

        Examples
        --------
        >>> con.drop_database("my_database")
        >>> con.drop_database("my_database", force=True)  # Won't fail if not exists
        """
        sql = sge.Drop(
            kind="DATABASE", exists=force, this=sg.table(name, catalog=catalog)
        ).sql(self.dialect)
        with self.begin() as cur:
            cur.execute(sql)

    def list_databases(self, *, like: str | None = None) -> list[str]:
        """Return the list of databases.

        Parameters
        ----------
        like
            A pattern in Python's regex format to filter returned database names.

        Returns
        -------
        list[str]
            The database names that match the pattern `like`.
        """
        query = "SHOW DATABASES"
        with self.begin() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        """List tables in SingleStoreDB database.

        Parameters
        ----------
        like
            SQL LIKE pattern to filter table names.
            Use '%' as wildcard, e.g., 'user_%' for tables starting with 'user_'
        database
            Database to list tables from. If None, uses current database.
            Can be a string database name or tuple (catalog, database)

        Returns
        -------
        list[str]
            List of table names in the specified database

        Examples
        --------
        >>> con.list_tables()
        ['users', 'orders', 'products']
        >>> con.list_tables(like="user_%")
        ['users', 'user_profiles']
        >>> con.list_tables(database="other_db")
        ['table1', 'table2']
        """
        from operator import itemgetter

        import sqlglot as sg
        import sqlglot.expressions as sge

        from ibis.backends.sql.compilers.base import TRUE, C

        if database is not None:
            table_loc = self._to_sqlglot_table(database)
        else:
            table_loc = sge.Table(
                db=sg.to_identifier(self.current_database, quoted=self.compiler.quoted),
                catalog=None,
            )

        conditions = [TRUE]

        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
        if table_loc.catalog or table_loc.db:
            conditions = [C.table_schema.eq(sge.convert(table_loc.sql("singlestore")))]

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql("singlestore")
        )

        with self.begin() as cur:
            cur.execute(sql)
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        """Get schema for a table in SingleStoreDB.

        Parameters
        ----------
        name
            Table name to get schema for
        catalog
            Catalog name (usually not used in SingleStoreDB)
        database
            Database name. If None, uses current database

        Returns
        -------
        Schema
            Ibis schema object with column names and types

        Examples
        --------
        >>> schema = con.get_schema("users")
        >>> print(schema)
        Schema:
          id: int64
          name: string
          email: string
          created_at: timestamp
        """
        import sqlglot as sg
        import sqlglot.expressions as sge

        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql("singlestore")  # Use singlestore dialect

        with self.begin() as cur:
            try:
                cur.execute(sge.Describe(this=table).sql("singlestore"))
            except Exception as e:
                # Handle table not found
                if "doesn't exist" in str(e) or "Table" in str(e):
                    raise com.TableNotFound(name) from e
                raise
            else:
                result = cur.fetchall()

        type_mapper = self.compiler.type_mapper
        fields = {
            name: type_mapper.from_string(type_string, nullable=is_nullable == "YES")
            for name, type_string, is_nullable, *_ in result
        }

        return sch.Schema(fields)

    @contextlib.contextmanager
    def begin(self) -> Generator[Any, None, None]:
        """Begin a transaction context for executing SQL commands.

        This method provides a cursor context manager that automatically
        handles transaction lifecycle including rollback on exceptions
        and proper cleanup.

        Yields
        ------
        Cursor
            SingleStoreDB cursor for executing SQL commands

        Examples
        --------
        >>> with con.begin() as cur:
        ...     cur.execute("SELECT COUNT(*) FROM users")
        ...     result = cur.fetchone()
        """
        con = self._client
        cur = con.cursor()
        autocommit = getattr(con, "autocommit", True)

        if not autocommit:
            con.begin()

        try:
            yield cur
        except Exception:
            if not autocommit and hasattr(con, "rollback"):
                con.rollback()
            raise
        else:
            if not autocommit and hasattr(con, "commit"):
                con.commit()
        finally:
            cur.close()

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments
        """

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params, **kwargs)

        schema = table.schema()

        with self._safe_raw_sql(sql) as cur:
            result = self._fetch_from_cursor(cur, schema)
        return expr.__pandas_result__(result)

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table in SingleStoreDB.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            Data to insert into the table. Can be an Ibis table expression,
            pandas DataFrame, PyArrow table, or Polars DataFrame/LazyFrame
        schema
            Schema for the table. If None, inferred from obj
        database
            Database to create the table in. If None, uses current database
        temp
            Create a temporary table
        overwrite
            Replace the table if it already exists

        Returns
        -------
        Table
            The created table expression

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        >>> table = con.create_table("my_table", df)
        >>> # Create with explicit schema
        >>> import ibis
        >>> schema = ibis.schema({"id": "int64", "name": "string"})
        >>> table = con.create_table("users", schema=schema)
        >>> # Create temporary table
        >>> temp_table = con.create_table("temp_data", df, temp=True)
        """
        import sqlglot as sg
        import sqlglot.expressions as sge

        import ibis
        import ibis.expr.operations as ops
        import ibis.expr.types as ir
        from ibis import util

        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite and not temp:
            # For non-temporary tables, use the rename strategy
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            # For temporary tables or non-overwrite, use the target name directly
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        # For temporary tables, don't include the database prefix as it's not allowed
        table_database = database if not temp else None
        table_expr = sg.table(temp_name, catalog=table_database, quoted=quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot_column_defs(dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE", this=target, properties=sge.Properties(expressions=properties)
        )

        this = sg.table(name, catalog=table_database, quoted=quoted)

        # Convert SQLGlot object to SQL string before execution
        with self.begin() as cur:
            if overwrite and temp:
                # For temporary tables with overwrite, drop the existing table first
                try:
                    cur.execute(
                        sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect)
                    )
                except Exception:
                    # Ignore errors if table doesn't exist
                    pass

            cur.execute(create_stmt.sql(dialect))
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite and not temp:
                # Only use rename strategy for non-temporary tables
                final_this = sg.table(name, catalog=database, quoted=quoted)
                cur.execute(
                    sge.Drop(kind="TABLE", this=final_this, exists=True).sql(dialect)
                )
                # Use ALTER TABLE ... RENAME TO syntax supported by SingleStoreDB
                # Extract just the table name (removing catalog/database prefixes and quotes)
                temp_table_name = temp_name
                if quoted:
                    temp_table_name = f"`{temp_name}`"
                final_table_name = name
                if quoted:
                    final_table_name = f"`{name}`"

                rename_sql = (
                    f"ALTER TABLE {temp_table_name} RENAME TO {final_table_name}"
                )
                cur.execute(rename_sql)

        if schema is None:
            return self.table(name, database=database if not temp else None)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(database=database if not temp else None),
        ).to_expr()

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table from the database.

        Parameters
        ----------
        name
            Table name to drop
        database
            Database name
        force
            Use IF EXISTS clause when dropping
        """
        drop_stmt = sge.Drop(
            kind="TABLE",
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            exists=force,
        )
        # Convert SQLGlot object to SQL string before execution
        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def _register_in_memory_table(self, op: Any) -> None:
        """Register an in-memory table in SingleStoreDB."""
        import sqlglot as sg
        import sqlglot.expressions as sge

        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "SingleStoreDB cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # Check for unsupported complex types
        for field_name, field_type in schema.items():
            if field_type.is_array() or field_type.is_struct() or field_type.is_map():
                raise com.UnsupportedBackendType(
                    f"SingleStoreDB does not support complex types like arrays, structs, or maps. "
                    f"Column '{field_name}' has type '{field_type}'"
                )

        name = op.name
        quoted = self.compiler.quoted
        dialect = self.dialect

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(dialect),
            ),
            properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
        )
        create_stmt_sql = create_stmt.sql(dialect)

        df = op.data.to_frame()
        df = df.replace(float("nan"), None)

        # Fix: Convert itertuples result to list for SingleStoreDB compatibility
        data = list(df.itertuples(index=False))
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                cur.executemany(sql, data)

    # TODO(kszucs): should make it an abstract method or remove the use of it
    # from .execute()
    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.raw_sql(*args, **kwargs) as result:
            yield result

    def _get_table_schema_from_describe(self, table_name: str) -> sch.Schema:
        """Get table schema using DESCRIBE and backend-specific type parsing."""
        from ibis.backends.singlestoredb.datatypes import SingleStoreDBType

        with self._safe_raw_sql(f"DESCRIBE {table_name}") as cur:
            rows = cur.fetchall()

        # Use backend-specific type parsing instead of generic ibis.dtype()
        types = []
        names = []
        for name, typ, *_ in rows:
            names.append(name)
            # Use SingleStoreDB-specific type parsing
            parsed_type = SingleStoreDBType.from_string(typ)
            types.append(parsed_type)

        return sch.Schema(dict(zip(names, types)))

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        """Execute a raw SQL query and return the cursor.

        Parameters
        ----------
        query
            SQL query string or SQLGlot expression to execute
        kwargs
            Additional parameters to pass to the query execution

        Returns
        -------
        Cursor
            Database cursor with query results

        Examples
        --------
        >>> cursor = con.raw_sql("SELECT * FROM users WHERE id = %s", (123,))
        >>> results = cursor.fetchall()
        >>> cursor.close()
        >>> # Using with context manager
        >>> with con.raw_sql("SHOW TABLES") as cursor:
        ...     tables = [row[0] for row in cursor.fetchall()]
        """
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        autocommit = con.get_autocommit()

        cursor = con.cursor()

        if not autocommit:
            con.begin()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            if not autocommit:
                con.rollback()
            cursor.close()
            raise
        else:
            if not autocommit:
                con.commit()
            return cursor

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get the schema of a query result."""
        import sqlglot as sg
        from sqlglot import expressions as sge

        from ibis import util
        from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData
        from ibis.backends.singlestoredb.datatypes import _type_from_cursor_info

        # First try to wrap the query directly without parsing
        # This avoids issues with sqlglot's SingleStore parser on complex queries
        sql = f"SELECT * FROM ({query}) AS {util.gen_name('query_schema')} LIMIT 0"

        try:
            with self.begin() as cur:
                cur.execute(sql)
                description = cur.description
        except Exception:
            # Fallback to the original parsing approach if direct wrapping fails
            try:
                # First try with SingleStore dialect
                parsed = sg.parse_one(query, dialect=self.dialect)
            except Exception:
                try:
                    # Fallback to MySQL dialect which SingleStore is based on
                    parsed = sg.parse_one(query, dialect="mysql")
                except Exception:
                    # Last resort - use generic SQL dialect
                    parsed = sg.parse_one(query, dialect="")

            # Use SQLGlot to properly construct the query
            sql = (
                sg.select(sge.Star())
                .from_(
                    parsed.subquery(
                        sg.to_identifier(
                            util.gen_name("query_schema"), quoted=self.compiler.quoted
                        )
                    )
                )
                .limit(0)
                .sql(self.dialect)
            )

            with self.begin() as cur:
                cur.execute(sql)
                description = cur.description

        names = []
        ibis_types = []
        for col_info in description:
            name = col_info[0]
            names.append(name)

            # Use the detailed cursor info for type conversion
            if len(col_info) >= 7:
                # Full cursor description available
                # SingleStoreDB uses 4-byte character encoding by default
                ibis_type = _type_from_cursor_info(
                    flags=col_info[7] if len(col_info) > 7 else 0,
                    type_code=col_info[1],
                    field_length=col_info[3],
                    scale=col_info[5],
                    multi_byte_maximum_length=4,  # Use 4 for SingleStoreDB's UTF8MB4 encoding
                )
            else:
                # Fallback for limited cursor info
                typename = SingleStoreDBPandasData._get_type_name(col_info[1])
                ibis_type = SingleStoreDBPandasData.convert_SingleStoreDB_type(typename)

            ibis_types.append(ibis_type)

        return sch.Schema(dict(zip(names, ibis_types)))

    @property
    def version(self) -> str:
        """Return the version of the SingleStoreDB server.

        Returns
        -------
        str
            The version string of the connected SingleStoreDB server.

        Examples
        --------
        >>> con.version  # doctest: +SKIP
        '8.7.10-bf633c1a54'
        """
        with self.begin() as cur:
            cur.execute("SELECT @@version")
            (version_string,) = cur.fetchone()
        return version_string

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
        database: str | None = None,
        driver: str | None = None,
        autocommit: bool = True,
        local_infile: bool = True,
        **kwargs,
    ) -> None:
        """Create an Ibis client connected to a SingleStoreDB database.

        Parameters
        ----------
        host : str, optional
            Hostname or URL
        user : str, optional
            Username
        password : str, optional
            Password
        port : int, optional
            Port number
        database : str, optional
            Database to connect to
        driver : str, optional
            Driver name: mysql, https, http
        autocommit : bool, default True
            Whether to autocommit transactions
        local_infile : bool, default True
            Enable LOAD DATA LOCAL INFILE support
        kwargs : dict, optional
            Additional keyword arguments passed to the underlying client
        """
        import singlestoredb as s2
        from singlestoredb.connection import build_params

        if driver:
            driver = driver.split("+", 1)[-1].replace("singlestoredb", "mysql")

        params = {
            k: v
            for k, v in dict(locals()).items()
            if k not in ("self",) and v is not None
        }

        self._original_connect_params = build_params(**params)

        self._client = s2.connect(**self._original_connect_params)

        return self._post_connect()

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table in SingleStoreDB.

        Parameters
        ----------
        old_name
            Current name of the table
        new_name
            New name for the table

        Examples
        --------
        >>> con.rename_table("old_table", "new_table")
        """
        old_name = self._quote_table_name(old_name)
        new_name = self._quote_table_name(new_name)
        with self.begin() as cur:
            cur.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")

    # Method removed - SingleStoreDB doesn't support catalogs

    def _quote_table_name(self, name: str) -> str:
        """Quote a table name for safe SQL usage.

        Parameters
        ----------
        name
            Table name to quote

        Returns
        -------
        str
            Quoted table name safe for SQL usage
        """
        import sqlglot as sg

        return sg.to_identifier(name, quoted=True).sql("singlestore")


def connect(
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    port: int | None = None,
    database: str | None = None,
    driver: str | None = None,
    autocommit: bool = True,
    local_infile: bool = True,
    **kwargs: Any,
) -> Backend:
    """Create an Ibis client connected to a SingleStoreDB database.

    Parameters
    ----------
    host : str, optional
        SingleStoreDB hostname or IP address
    user : str, optional
        Username for authentication
    password : str, optional
        Password for authentication
    port : int, optional
        Port number (default 3306)
    database : str, optional
        Database name to connect to
    driver : str, optional
        Driver name: mysql, https, http
    autocommit : bool, default True
        Whether to autocommit transactions
    local_infile : bool, default True
        Enable LOAD DATA LOCAL INFILE support
    kwargs
        Additional connection parameters:
        - local_infile: Enable LOCAL INFILE capability (default 0)
        - charset: Character set (default utf8mb4)
        - ssl_disabled: Disable SSL connection
        - connect_timeout: Connection timeout in seconds
        - read_timeout: Read timeout in seconds
        - write_timeout: Write timeout in seconds
        See SingleStoreDB Python client documentation for more options.

    Returns
    -------
    Backend
        An Ibis SingleStoreDB backend instance

    Examples
    --------
    Basic connection:

    >>> import ibis
    >>> con = ibis.singlestoredb.connect(
    ...     host="localhost", user="root", password="password", database="my_database"
    ... )

    Connection with additional options:

    >>> con = ibis.singlestoredb.connect(
    ...     host="singlestore.example.com",
    ...     port=3306,
    ...     user="app_user",
    ...     password="secret",
    ...     database="production",
    ...     autocommit=True,
    ...     connect_timeout=30,
    ... )

    Using connection string (alternative method):

    >>> con = ibis.connect("singlestoredb://user:password@host:port/database")
    """
    backend = Backend()
    backend.do_connect(
        host=host,
        user=user,
        password=password,
        port=port,
        database=database,
        driver=driver,
        autocommit=autocommit,
        local_infile=local_infile,
        **kwargs,
    )
    return backend
