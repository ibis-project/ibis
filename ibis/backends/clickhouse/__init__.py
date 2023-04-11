from __future__ import annotations

import ast
import contextlib
import json
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import clickhouse_driver
import sqlalchemy as sa
import sqlglot as sg
import toolz

import ibis
import ibis.common.exceptions as com
import ibis.config
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import BaseBackend
from ibis.backends.clickhouse.compiler import translate
from ibis.backends.clickhouse.datatypes import parse, serialize

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

_default_compression: str | bool

try:
    import clickhouse_cityhash  # noqa: F401
    import lz4  # noqa: F401

    _default_compression = 'lz4'
except ImportError:
    _default_compression = False


class ClickhouseTable(ir.Table):
    """References a physical table in Clickhouse."""

    @property
    def _client(self):
        return self.op().source

    @property
    def name(self):
        return self.op().name

    def insert(self, obj, **kwargs):
        import pandas as pd

        from ibis.backends.clickhouse.identifiers import quote_identifier

        schema = self.schema()

        assert isinstance(obj, pd.DataFrame)
        assert set(schema.names) >= set(obj.columns)

        columns = ", ".join(map(quote_identifier, obj.columns))
        query = f"INSERT INTO {self.name} ({columns}) VALUES"

        settings = kwargs.pop("settings", {})
        settings["use_numpy"] = True
        return self._client._client.insert_dataframe(
            query,
            obj,
            settings=settings,
            **kwargs,
        )


class Backend(BaseBackend):
    name = 'clickhouse'

    class Options(ibis.config.Config):
        """Clickhouse options.

        Attributes
        ----------
        temp_db : str
            Database to use for temporary objects.
        bool_type : str
            Type to use for boolean columns
        """

        temp_db: str = "__ibis_tmp"
        bool_type: Literal["Bool", "UInt8", "Int8"] = "Bool"

    def __init__(self, *args, external_tables=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_tables = toolz.valmap(
            lambda v: ibis.memtable(v).op(), external_tables or {}
        )

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self._external_tables[op.name] = op

    def _log(self, sql: str) -> None:
        """Log the SQL, usually to the standard output.

        This method can be implemented by subclasses. The logging
        happens when `ibis.options.verbose` is `True`.
        """
        util.log(sql)

    def sql(self, query: str, schema=None) -> ir.Table:
        if schema is None:
            schema = self._get_schema_using_query(query)
        return ops.SQLQueryResult(query, ibis.schema(schema), self).to_expr()

    def _from_url(self, url: str, **kwargs) -> BaseBackend:
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments

        Returns
        -------
        BaseBackend
            A backend instance
        """
        url = sa.engine.make_url(url)

        kwargs = toolz.merge(
            {
                name: value
                for name in ("host", "port", "database", "password")
                if (value := getattr(url, name, None))
            },
            kwargs,
        )
        if username := url.username:
            kwargs["user"] = username

        kwargs.update(url.query)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["secure"] = bool(ast.literal_eval(kwargs["secure"]))

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
        external_tables
            External tables that can be used in a query.
        kwargs
            Client specific keyword arguments

        Examples
        --------
        >>> import ibis
        >>> import os
        >>> clickhouse_host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        >>> clickhouse_port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        >>> client = ibis.clickhouse.connect(host=clickhouse_host,  port=clickhouse_port)
        >>> client  # doctest: +ELLIPSIS
        <ibis.clickhouse.client.ClickhouseClient object at 0x...>
        """
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
        self._external_tables = toolz.valmap(
            lambda v: ibis.memtable(v).op(), external_tables or {}
        )

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
        """Merge registered external tables with any new external tables."""
        external_tables_list = []
        for name, obj in toolz.merge(
            self._external_tables,
            toolz.valmap(lambda v: ibis.memtable(v).op(), external_tables or {}),
        ).items():
            if not (schema := obj.schema):
                raise TypeError(f'Schema is empty for external table {name}')

            df = obj.data.to_frame()
            structure = list(
                zip(
                    schema.names,
                    map(
                        serialize,
                        (
                            # unwrap nested structures because clickhouse does
                            # not accept nullable arrays, maps or structs
                            typ.copy(nullable=not typ.is_nested())
                            for typ in schema.types
                        ),
                    ),
                )
            )
            data = dict(name=name, data=df.to_dict("records"), structure=structure)
            external_tables_list.append(data)
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

    def _register_in_memory_tables(self, expr: ir.TableExpr):
        for memtable in an.find_memtables(expr.op()):
            self._register_in_memory_table(memtable)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return an iterator of pyarrow record batches.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.

        Returns
        -------
        results
            RecordBatchReader
        """
        pa = self._import_pyarrow()

        schema = expr.as_table().schema()
        array_type = schema.as_struct().to_pyarrow()
        batches = (
            pa.RecordBatch.from_struct_array(pa.array(batch, type=array_type))
            for batch in self._cursor_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size
            )
        )

        return pa.ipc.RecordBatchReader.from_batches(schema.to_pyarrow(), batches)

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
    ) -> Iterable[list]:
        self._register_in_memory_tables(expr)
        sql = self.compile(expr, limit=limit, params=params)
        cursor = self.raw_sql(sql)
        try:
            while batch := cursor.fetchmany(chunk_size):
                yield batch
        finally:
            cursor.close()

    def execute(
        self,
        expr: ir.Expr,
        limit: str | None = 'default',
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""
        self._register_in_memory_tables(expr)
        table_expr = expr.as_table()
        sql = self.compile(table_expr, limit=limit, **kwargs)
        self._log(sql)
        result = self.fetch_from_cursor(
            self.raw_sql(sql, external_tables=external_tables),
            table_expr.schema(),
        )
        if isinstance(expr, ir.Scalar):
            return result.iloc[0, 0]
        elif isinstance(expr, ir.Column):
            return result.iloc[:, 0]
        else:
            return result

    def compile(self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any):
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = translate(table_expr.op(), params=params)
        assert not isinstance(sql, sg.exp.Subquery)

        if isinstance(sql, sg.exp.Table):
            sql = sg.select("*").from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql.sql(dialect="clickhouse", pretty=True)

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        return str(self.compile(expr, **kwargs))

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        database
            Database name

        Returns
        -------
        Table
            Table expression
        """
        schema = self.get_schema(name, database=database)
        qname = self._fully_qualified_name(name, database)
        return ClickhouseTable(ops.DatabaseTable(qname, schema, self))

    def raw_sql(
        self,
        query: str,
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **_,
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

    def fetch_from_cursor(self, cursor, schema):
        import pandas as pd

        df = pd.DataFrame.from_records(iter(cursor), columns=schema.names)
        return schema.apply_to(df)

    def close(self):
        """Close Clickhouse connection and drop any temporary objects."""
        self._client.disconnect()
        self.con.close()

    def _fully_qualified_name(self, name: str, database: str | None) -> str:
        return sg.table(name, db=database or self.current_database or None).sql(
            dialect="clickhouse"
        )

    def get_schema(self, table_name: str, database: str | None = None) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
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

        return sch.Schema(dict(zip(column_names, map(parse, types))))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        [(raw_plans,)] = self._client.execute(
            f"EXPLAIN json = 1, description = 0, header = 1 {query}"
        )
        [plan] = json.loads(raw_plans)
        return sch.Schema(
            {field["Name"]: parse(field["Type"]) for field in plan["Plan"]["Header"]}
        )

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        from ibis.backends.clickhouse.compiler.values import translate_val

        return operation in translate_val.registry

    def create_database(
        self, name: str, *, force: bool = False, engine: str = "Atomic"
    ) -> None:
        if_not_exists = "IF NOT EXISTS " * force
        self.raw_sql(f"CREATE DATABASE {if_not_exists}{name} ENGINE = {engine}")

    def drop_database(self, name: str, *, force: bool = False) -> None:
        if_exists = "IF EXISTS " * force
        self.raw_sql(f"DROP DATABASE {if_exists}{name}")

    def truncate_table(self, name: str, database: str | None = None) -> None:
        ident = self._fully_qualified_name(name, database)
        self.raw_sql(f"TRUNCATE TABLE {ident}")

    def drop_table(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        ident = self._fully_qualified_name(name, database)
        self.raw_sql(f"DROP TABLE {'IF EXISTS ' * force}{ident}")

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        # backend specific arguments
        engine: str,
        order_by: Iterable[str] | None = None,
        partition_by: Iterable[str] | None = None,
        sample_by: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> ir.Table:
        """Create a table in a ClickHouse database.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            Optional data to create the table with
        schema
            Optional names and types of the table
        database
            Database to create the table in
        temp
            Create a temporary table. This is not yet supported, and exists for
            API compatibility.
        overwrite
            Whether to overwrite the table
        engine
            The table engine to use. See [ClickHouse's `CREATE TABLE` documentation](https://clickhouse.com/docs/en/sql-reference/statements/create/table)
            for specifics.
        order_by
            String column names to order by. Required for some table engines like `MergeTree`.
        partition_by
            String column names to partition by
        sample_by
            String column names to sample by
        settings
            Key-value pairs of settings for table creation

        Returns
        -------
        Table
            The new table
        """
        if temp:
            raise com.IbisError("ClickHouse temporary tables are not yet supported")

        tmp = "TEMPORARY " * temp
        replace = "OR REPLACE " * overwrite
        code = (
            f"CREATE {replace}{tmp}TABLE {self._fully_qualified_name(name, database)}"
        )

        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")

        if schema is not None:
            serialized_schema = ", ".join(
                f"`{name}` {serialize(typ)}" for name, typ in schema.items()
            )
            code += f" ({serialized_schema})"

        if obj is not None and not isinstance(obj, ir.Expr):
            obj = ibis.memtable(obj, schema=schema)

        code += f" ENGINE = {engine}"

        if order_by is not None:
            code += f" ORDER BY {', '.join(util.promote_list(order_by))}"

        if partition_by is not None:
            code += f" PARTITION BY {', '.join(util.promote_list(partition_by))}"

        if sample_by is not None:
            code += f" SAMPLE BY {sample_by}"

        if settings:
            kvs = ", ".join(f"{name}={value!r}" for name, value in settings.items())
            code += f" SETTINGS {kvs}"

        if obj is not None:
            self._register_in_memory_tables(obj)
            query = self.compile(obj)
            code += f" AS {query}"

        self.raw_sql(code)
        return self.table(name, database=database)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        qualname = self._fully_qualified_name(name, database)
        replace = "OR REPLACE " * overwrite
        query = self.compile(obj)
        code = f"CREATE {replace}VIEW {qualname} AS {query}"
        self.raw_sql(code)
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        name = self._fully_qualified_name(name, database)
        if_exists = "IF EXISTS " * force
        self.raw_sql(f"DROP VIEW {if_exists}{name}")
