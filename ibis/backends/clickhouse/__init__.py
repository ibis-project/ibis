from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import clickhouse_driver
import toolz

import ibis
import ibis.config
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.clickhouse.client import ClickhouseTable, fully_qualified_re
from ibis.backends.clickhouse.compiler import ClickhouseCompiler
from ibis.backends.clickhouse.datatypes import parse, serialize
from ibis.config import options

if TYPE_CHECKING:
    import pandas as pd

_default_compression: str | bool

try:
    import clickhouse_cityhash  # noqa: F401
    import lz4  # noqa: F401

    _default_compression = 'lz4'
except ImportError:
    _default_compression = False


class Backend(BaseSQLBackend):
    name = 'clickhouse'
    # for now map clickhouse to mysql so that _something_ works
    _sqlglot_dialect = "mysql"
    table_expr_class = ClickhouseTable
    compiler = ClickhouseCompiler

    class Options(ibis.config.Config):
        """Clickhouse options.

        Attributes
        ----------
        temp_db : str
            Database to use for temporary objects.
        """

        temp_db: str = "__ibis_tmp"

    def __init__(self, *args, external_tables=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_tables = external_tables or {}

    def _register_in_memory_table(self, table_op):
        self._external_tables[table_op.name] = table_op.data.to_frame()

    def do_connect(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "default",
        user: str = "default",
        password: str = "",
        client_name: str = "ibis",
        compression: (
            Literal["lz4", "lz4hc", "quicklz", "zstd"] | bool
        ) = _default_compression,
        external_tables=None,
        **kwargs: Any,
    ):
        """Create a ClickHouse client for use with Ibis.

        Parameters
        ----------
        host
            Host name of the clickhouse server
        port
            Clickhouse server's  port
        database
            Default database when executing queries
        user
            User to authenticate with
        password
            Password to authenticate with
        client_name
            Name of client that wil appear in clickhouse server logs
        compression
            Whether or not to use compression.
            Default is `'lz4'` if installed else False.
            True is equivalent to `'lz4'`.

        Examples
        --------
        >>> import ibis
        >>> import os
        >>> clickhouse_host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        >>> clickhouse_port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        >>> client = ibis.clickhouse.connect(host=clickhouse_host,  port=clickhouse_port)
        >>> client  # doctest: +ELLIPSIS
        <ibis.clickhouse.client.ClickhouseClient object at 0x...>
        """  # noqa: E501
        options = dict(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            client_name=client_name,
            compression=compression,
            **kwargs,
        )
        # We use the client for any ibis-native queries for efficiency
        self._client = clickhouse_driver.Client(**options)
        # We use the dbapi for `raw_sql` calls to provide a cursor object.
        # This won't start a connection until `cursor` is called, so in
        # the common case this is cheap.
        self.con = clickhouse_driver.dbapi.connect(**options)
        self._external_tables = external_tables or {}

    @property
    def version(self) -> str:
        self._client.connection.force_connect()
        try:
            info = self._client.connection.server_info
        finally:
            self._client.connection.disconnect()

        return f'{info.version_major}.{info.version_minor}.{info.revision}'

    @property
    def current_database(self):
        return self._client.connection.database

    def list_databases(self, like=None):
        data, _ = self._client_execute('SELECT name FROM system.databases')
        # in theory this should never be empty
        if not data:  # pragma: no cover
            return []
        databases = list(data[0])
        return self._filter_with_like(databases, like)

    def list_tables(self, like=None, database=None):
        if database is not None:
            query = f"SHOW TABLES FROM `{database}`"
        else:
            query = "SHOW TABLES"
        data, _ = self._client_execute(query)
        if not data:
            return []
        tables = list(data[0])
        return self._filter_with_like(tables, like)

    def _normalize_external_tables(self, external_tables=None):
        """Merge registered external tables with any new external tables, and
        process them to be passed to clickhouse_driver."""
        import pandas as pd

        external_tables_list = []
        if external_tables is None:
            external_tables = {}
        for name, df in toolz.merge(self._external_tables, external_tables).items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError('External table is not an instance of pandas dataframe')
            schema = sch.infer(df)
            external_tables_list.append(
                {
                    "name": name,
                    "data": df.to_dict("records"),
                    "structure": list(zip(schema.names, map(serialize, schema.types))),
                }
            )
        return external_tables_list

    def _client_execute(self, query, external_tables=None):
        external_tables = self._normalize_external_tables(external_tables)
        ibis.util.log(query)
        with self._client as con:
            return con.execute(
                query,
                columnar=True,
                with_column_types=True,
                external_tables=external_tables,
            )

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
    ) -> Iterable[list]:
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        cursor = self.raw_sql(str(sql))
        try:
            while batch := cursor.fetchmany(chunk_size):
                yield batch
        finally:
            cursor.close()

    def raw_sql(
        self,
        query: str,
        external_tables: Mapping[str, pd.DataFrame] | None = None,
    ) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        external_tables
            Mapping of table name to pandas DataFrames providing
            external datasources for the query

        Returns
        -------
        Cursor
            Clickhouse cursor
        """
        cursor = self.con.cursor()
        for kws in self._normalize_external_tables(external_tables):
            cursor.set_external_table(**kws)

        ibis.util.log(query)
        cursor.execute(query)
        return cursor

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        results = self._client_execute(*args, **kwargs)
        yield results

    def fetch_from_cursor(self, cursor, schema):
        import pandas as pd

        data, _ = cursor
        names = schema.names
        if not data:
            df = pd.DataFrame([], columns=names)
        else:
            df = pd.DataFrame.from_dict(dict(zip(names, data)))
        return schema.apply_to(df)

    def close(self):
        """Close Clickhouse connection and drop any temporary objects."""
        self._client.disconnect()
        self.con.close()

    def _fully_qualified_name(self, name, database):
        if fully_qualified_re.search(name):
            return name

        database = database or self.current_database
        return f'{database}.`{name}`'

    def get_schema(
        self,
        table_name: str,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May be fully qualified
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)
        (column_names, types, *_), *_ = self._client_execute(
            f"DESCRIBE {qualified_name}"
        )

        return sch.Schema.from_tuples(zip(column_names, map(parse, types)))

    def _ensure_temp_db_exists(self):
        name = (options.clickhouse.temp_db,)
        if name not in self.list_databases():
            self.create_database(name, force=True)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        [(raw_plans,)] = self._client.execute(
            f"EXPLAIN json = 1, description = 0, header = 1 {query}"
        )
        [plan] = json.loads(raw_plans)
        fields = [
            (field["Name"], parse(field["Type"])) for field in plan["Plan"]["Header"]
        ]
        return sch.Schema.from_tuples(fields)

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return f'{cmd} {qualified_name}'
