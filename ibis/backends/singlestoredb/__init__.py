"""The SingleStoreDB backend."""

from __future__ import annotations

import contextlib
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

if TYPE_CHECKING:
    from collections.abc import Generator

import ibis.common.exceptions as com
import ibis.expr.schema as sch
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend

if TYPE_CHECKING:
    from urllib.parse import ParseResult


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "singlestoredb"
    supports_create_or_replace = False
    supports_temporary_tables = True

    # SingleStoreDB inherits MySQL protocol compatibility
    _connect_string_template = (
        "singlestoredb://{{user}}:{{password}}@{{host}}:{{port}}/{{database}}"
    )

    @property
    def compiler(self):
        """Return the SQL compiler for SingleStoreDB."""
        from ibis.backends.sql.compilers.singlestoredb import compiler

        return compiler

    @property
    def con(self):
        """Return the database connection for compatibility with base class."""
        return self._client

    @property
    def current_database(self) -> str:
        """Return the current database name."""
        with self._safe_raw_sql("SELECT DATABASE()") as cur:
            (database,) = cur.fetchone()
        return database

    def do_connect(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "",
        port: int = 3306,
        database: str = "",
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to a SingleStoreDB database.

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
        kwargs
            Additional connection parameters
        """
        # Use SingleStoreDB client exclusively
        import singlestoredb as s2

        self._client = s2.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database=database,
            autocommit=kwargs.pop("autocommit", True),
            local_infile=kwargs.pop("local_infile", 0),
            **kwargs,
        )

    @classmethod
    def _from_url(cls, url: ParseResult, **kwargs) -> Backend:
        """Create a SingleStoreDB backend from a connection URL."""
        database = url.path[1:] if url.path and len(url.path) > 1 else ""

        backend = cls()
        backend.do_connect(
            host=url.hostname or "localhost",
            port=url.port or 3306,
            user=url.username or "root",
            password=unquote_plus(url.password or ""),
            database=database,
            **kwargs,
        )
        return backend

    def create_database(self, name: str, force: bool = False) -> None:
        """Create a database in SingleStoreDB.

        Parameters
        ----------
        name
            Database name to create
        force
            If True, use CREATE DATABASE IF NOT EXISTS to avoid errors
            if the database already exists

        Examples
        --------
        >>> con.create_database("my_database")
        >>> con.create_database("existing_db", force=True)  # Won't fail if exists
        """
        if_not_exists = "IF NOT EXISTS " * force
        with self._safe_raw_sql(f"CREATE DATABASE {if_not_exists}{name}"):
            pass

    def drop_database(self, name: str, force: bool = False) -> None:
        """Drop a database in SingleStoreDB.

        Parameters
        ----------
        name
            Database name to drop
        force
            If True, use DROP DATABASE IF EXISTS to avoid errors
            if the database doesn't exist

        Examples
        --------
        >>> con.drop_database("old_database")
        >>> con.drop_database("maybe_exists", force=True)  # Won't fail if missing
        """
        if_exists = "IF EXISTS " * force
        with self._safe_raw_sql(f"DROP DATABASE {if_exists}{name}"):
            pass

    def list_databases(self, like: str | None = None) -> list[str]:
        """List databases in the SingleStoreDB cluster.

        Parameters
        ----------
        like
            SQL LIKE pattern to filter database names.
            Use '%' as wildcard, e.g., 'test_%' for databases starting with 'test_'

        Returns
        -------
        list[str]
            List of database names

        Examples
        --------
        >>> con.list_databases()
        ['information_schema', 'mysql', 'my_app_db', 'test_db']
        >>> con.list_databases(like="test_%")
        ['test_db', 'test_staging']
        """
        query = "SHOW DATABASES"
        if like is not None:
            query += f" LIKE '{like}'"

        with self._safe_raw_sql(query) as cur:
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
            conditions = [C.table_schema.eq(sge.convert(table_loc.sql("mysql")))]

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql("mysql")
        )

        with self._safe_raw_sql(sql) as cur:
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
        ).sql("mysql")  # Use mysql dialect for compatibility

        with self.begin() as cur:
            try:
                cur.execute(sge.Describe(this=table).sql("mysql"))
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
        handles cleanup. Use this for executing raw SQL commands.

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
        cursor = self._client.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def create_table(
        self,
        name: str,
        /,
        obj: Any | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in SingleStoreDB."""
        import sqlglot as sg
        import sqlglot.expressions as sge

        import ibis
        import ibis.expr.operations as ops
        import ibis.expr.types as ir
        from ibis import util
        from ibis.backends.sql.compilers.base import RenameTable

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

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        table_expr = sg.table(temp_name, catalog=database, quoted=quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot_column_defs(dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE", this=target, properties=sge.Properties(expressions=properties)
        )

        this = sg.table(name, catalog=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite:
                cur.execute(sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect))
                cur.execute(
                    sge.Alter(
                        kind="TABLE",
                        this=table_expr,
                        exists=True,
                        actions=[RenameTable(this=this)],
                    ).sql(dialect)
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

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
        # nan can not be used with SingleStoreDB like MySQL
        df = df.replace(float("nan"), None)

        data = df.itertuples(index=False)
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                cur.executemany(sql, data)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str, *args, **kwargs) -> Generator[Any, None, None]:
        """Execute raw SQL with proper error handling."""
        cursor = self._client.cursor()
        try:
            cursor.execute(query, *args, **kwargs)
            yield cursor
        except Exception as e:
            # Convert database-specific exceptions to Ibis exceptions
            if hasattr(e, "args") and len(e.args) > 1:
                errno, msg = e.args[:2]
                if errno == 1050:  # Table already exists
                    raise com.IntegrityError(msg)
                elif errno == 1146:  # Table doesn't exist
                    raise com.RelationError(msg)
                elif errno in (1054, 1064):  # Bad field name or syntax error
                    raise com.ExpressionError(msg)
            raise
        finally:
            cursor.close()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get the schema of a query result."""
        from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData
        from ibis.backends.singlestoredb.datatypes import _type_from_cursor_info

        with self.begin() as cur:
            cur.execute(f"({query}) LIMIT 0")
            description = cur.description

        names = []
        ibis_types = []
        for col_info in description:
            name = col_info[0]
            names.append(name)

            # Use the detailed cursor info for type conversion
            if len(col_info) >= 7:
                # Full cursor description available
                ibis_type = _type_from_cursor_info(
                    flags=col_info[7] if len(col_info) > 7 else 0,
                    type_code=col_info[1],
                    field_length=col_info[3],
                    scale=col_info[5],
                    multi_byte_maximum_length=1,  # Default for most cases
                )
            else:
                # Fallback for limited cursor info
                typename = SingleStoreDBPandasData._get_type_name(col_info[1])
                ibis_type = SingleStoreDBPandasData.convert_SingleStoreDB_type(typename)

            ibis_types.append(ibis_type)

        return sch.Schema(dict(zip(names, ibis_types)))

    @cached_property
    def version(self) -> str:
        """Return the SingleStoreDB server version.

        Returns
        -------
        str
            SingleStoreDB server version string

        Examples
        --------
        >>> con.version
        'SingleStoreDB 8.7.10'
        """
        with self._safe_raw_sql("SELECT @@version") as cur:
            (version_string,) = cur.fetchone()
        return version_string


def connect(
    host: str = "localhost",
    user: str = "root",
    password: str = "",
    port: int = 3306,
    database: str = "",
    **kwargs: Any,
) -> Backend:
    """Create an Ibis client connected to a SingleStoreDB database.

    Parameters
    ----------
    host
        SingleStoreDB hostname or IP address
    user
        Username for authentication
    password
        Password for authentication
    port
        Port number (default 3306)
    database
        Database name to connect to
    kwargs
        Additional connection parameters:
        - autocommit: Enable autocommit mode (default True)
        - local_infile: Enable LOCAL INFILE capability (default 0)
        - charset: Character set (default utf8mb4)
        - ssl_disabled: Disable SSL connection
        - connect_timeout: Connection timeout in seconds
        - read_timeout: Read timeout in seconds
        - write_timeout: Write timeout in seconds

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
        host=host, user=user, password=password, port=port, database=database, **kwargs
    )
    return backend
