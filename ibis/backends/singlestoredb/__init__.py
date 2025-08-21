"""The SingleStoreDB backend."""

from __future__ import annotations

import contextlib
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

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

        return compiler.with_params(
            default_schema=self.current_database, quoted=self.quoted
        )

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
        try:
            # Try SingleStoreDB client first
            import singlestoredb as s2

            self._client = s2.connect(
                host=host,
                user=user,
                password=password,
                port=port,
                database=database,
                autocommit=True,
                local_infile=kwargs.pop("local_infile", 0),
                **kwargs,
            )
        except ImportError:
            # Fall back to MySQLdb for compatibility
            import MySQLdb

            self._client = MySQLdb.connect(
                host=host,
                user=user,
                passwd=password,
                port=port,
                db=database,
                autocommit=True,
                local_infile=kwargs.pop("local_infile", 0),
                **kwargs,
            )

    @classmethod
    def _from_url(cls, url: ParseResult, **kwargs) -> Backend:
        """Create a SingleStoreDB backend from a connection URL."""
        database = url.path[1:] if url.path and len(url.path) > 1 else ""

        return cls.do_connect(
            host=url.hostname or "localhost",
            port=url.port or 3306,
            user=url.username or "root",
            password=unquote_plus(url.password or ""),
            database=database,
            **kwargs,
        )

    def create_database(self, name: str, force: bool = False) -> None:
        """Create a database in SingleStoreDB."""
        if_not_exists = "IF NOT EXISTS " * force
        with self._safe_raw_sql(f"CREATE DATABASE {if_not_exists}{name}"):
            pass

    def drop_database(self, name: str, force: bool = False) -> None:
        """Drop a database in SingleStoreDB."""
        if_exists = "IF EXISTS " * force
        with self._safe_raw_sql(f"DROP DATABASE {if_exists}{name}"):
            pass

    def list_databases(self, like: str | None = None) -> list[str]:
        """List databases in the SingleStoreDB cluster."""
        query = "SHOW DATABASES"
        if like is not None:
            query += f" LIKE '{like}'"

        with self._safe_raw_sql(query) as cur:
            return [row[0] for row in cur.fetchall()]

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str, *args, **kwargs):
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
        """Return the SingleStoreDB server version."""
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

    Returns
    -------
    Backend
        An Ibis SingleStoreDB backend instance

    Examples
    --------
    >>> import ibis
    >>> con = ibis.singlestoredb.connect(host="localhost", database="test")
    >>> con.list_tables()
    []
    """
    backend = Backend()
    backend.do_connect(
        host=host, user=user, password=password, port=port, database=database, **kwargs
    )
    return backend
