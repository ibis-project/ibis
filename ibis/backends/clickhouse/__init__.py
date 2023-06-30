from __future__ import annotations

import ast
import json
from contextlib import closing, suppress
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping

import clickhouse_connect as cc
import pyarrow as pa
import sqlalchemy as sa
import sqlglot as sg
import toolz
from clickhouse_connect.driver.external import ExternalData

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
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    import pandas as pd

    from ibis.common.typing import SupportsSchema


def _to_memtable(v):
    return ibis.memtable(v).op() if not isinstance(v, ops.InMemoryTable) else v


class ClickhouseTable(ir.Table):
    """References a physical table in Clickhouse."""

    @property
    def _client(self):
        return self.op().source

    @property
    def name(self):
        return self.op().name

    def insert(self, obj, settings: Mapping[str, Any] | None = None, **kwargs):
        import pandas as pd

        if not isinstance(obj, pd.DataFrame):
            raise com.IbisError(
                f"Invalid input type {type(obj)}; only pandas DataFrames are accepted as input"
            )

        return self._client.con.insert_df(self.name, obj, settings=settings, **kwargs)


class Backend(BaseBackend):
    name = "clickhouse"

    # ClickHouse itself does, but the client driver does not
    supports_temporary_tables = False

    class Options(ibis.config.Config):
        """Clickhouse options.

        Attributes
        ----------
        bool_type : str
            Type to use for boolean columns
        """

        bool_type: Literal["Bool", "UInt8", "Int8"] = "Bool"

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        util.log(sql)

    def sql(
        self,
        query: str,
        schema: SupportsSchema | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        query = self._transpile_sql(query, dialect=dialect)
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
        with suppress(KeyError):
            kwargs["secure"] = bool(ast.literal_eval(kwargs["secure"]))

    def do_connect(
        self,
        host: str = "localhost",
        port: int | None = None,
        database: str = "default",
        user: str = "default",
        password: str = "",
        client_name: str = "ibis",
        secure: bool | None = None,
        compression: str | bool = True,
        **kwargs: Any,
    ):
        """Create a ClickHouse client for use with Ibis.

        Parameters
        ----------
        host
            Host name of the clickhouse server
        port
            ClickHouse HTTP server's port. If not passed, the value depends on
            whether `secure` is `True` or `False`.
        database
            Default database when executing queries
        user
            User to authenticate with
        password
            Password to authenticate with
        client_name
            Name of client that will appear in clickhouse server logs
        secure
            Whether or not to use an authenticated endpoint
        compression
            The kind of compression to use for requests. See
            https://clickhouse.com/docs/en/integrations/python#compression for
            more information.
        kwargs
            Client specific keyword arguments

        Examples
        --------
        >>> import ibis
        >>> client = ibis.clickhouse.connect()
        >>> client
        <ibis.clickhouse.client.ClickhouseClient object at 0x...>
        """
        self.con = cc.get_client(
            host=host,
            # 8123 is the default http port 443 is https
            port=port if port is not None else 443 if secure else 8123,
            database=database,
            user=user,
            password=password,
            client_name=client_name,
            query_limit=0,
            compress=compression,
            **kwargs,
        )

    @property
    def version(self) -> str:
        return self.con.server_version

    @property
    def current_database(self) -> str:
        return self.con.database

    def list_databases(self, like: str | None = None) -> list[str]:
        with closing(self.raw_sql("SELECT name FROM system.databases")) as result:
            results = result.result_columns

        if results:
            (databases,) = results
        else:
            databases = []
        return self._filter_with_like(databases, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SHOW TABLES" + (f" FROM `{database}`" * (database is not None))
        with closing(self.raw_sql(query)) as result:
            results = result.result_columns

        if results:
            (tables,) = results
        else:
            tables = []
        return self._filter_with_like(tables, like)

    def _normalize_external_tables(self, external_tables=None) -> ExternalData | None:
        """Merge registered external tables with any new external tables."""
        external_data = ExternalData()
        n = 0
        for name, obj in (external_tables or {}).items():
            n += 1
            if not (schema := obj.schema):
                raise TypeError(f'Schema is empty for external table {name}')

            structure = [
                f"{name} {serialize(typ.copy(nullable=not typ.is_nested()))}"
                for name, typ in schema.items()
            ]
            external_data.add_file(
                file_name=name,
                data=obj.data.to_pyarrow_bytes(schema=schema),
                structure=structure,
                fmt="Arrow",
            )
        if not n:
            return None
        return external_data

    def _collect_in_memory_tables(
        self, expr: ir.TableExpr | None, *, external_tables: Mapping | None = None
    ):
        return toolz.merge(
            (
                {op.name: op for op in an.find_memtables(expr.op())}
                if expr is not None
                else {}
            ),
            external_tables or {},
        )

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        external_tables: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        # we convert to batches first to avoid a bunch of rigmarole resulting
        # from the following rough edges
        #
        # 1. clickhouse's awkward
        #    client-settings-are-permissioned-on-the-server "feature"
        # 2. the bizarre conversion of `DateTime64` without scale to arrow
        #    uint32 inside of clickhouse
        # 3. the fact that uint32 cannot be cast to pa.timestamp without first
        #    casting it to int64
        #
        # the extra code to make this dance work without first converting to
        # record batches isn't worth it without some benchmarking
        with self.to_pyarrow_batches(
            expr=expr,
            params=params,
            limit=limit,
            external_tables=external_tables,
            **kwargs,
        ) as reader:
            t = reader.read_all()

        if isinstance(expr, ir.Scalar):
            return t[0][0]
        elif isinstance(expr, ir.Column):
            return t[0]
        else:
            return t

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Scalar, Any] | None = None,
        external_tables: Mapping[str, Any] | None = None,
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
        external_tables
            External data
        chunk_size
            Maximum number of row to return in a single chunk

        Returns
        -------
        results
            RecordBatchReader

        Notes
        -----
        There are a variety of ways to implement clickhouse -> record batches.

        1. FORMAT ArrowStream -> record batches via raw_query
           This has the same type conversion problem(s) as `to_pyarrow`.
           It's harder to address due to lack of `cast` on `RecordBatch`.
           However, this is a ClickHouse problem: we should be able to get
           string data out without a bunch of settings/permissions rigmarole.
        2. Native -> Python objects -> pyarrow batches
           This is what is implemented, using `query_column_block_stream`.
        3. Native -> Python objects -> DataFrame chunks -> pyarrow batches
           This is not implemented because it adds an unnecessary pandas step in
           between Python object -> arrow. We can go directly to record batches
           without pandas in the middle.
        """
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params)

        external_data = self._normalize_external_tables(
            self._collect_in_memory_tables(expr, external_tables=external_tables)
        )

        def batcher(sql: str, *, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
            settings = {}

            # readonly != 1 means that the server setting is writable
            if self.con.server_settings["max_block_size"].readonly != 1:
                settings["max_block_size"] = chunk_size
            with self.con.query_column_block_stream(
                sql, external_data=external_data, settings=settings
            ) as blocks:
                yield from map(
                    partial(pa.RecordBatch.from_arrays, schema=schema), blocks
                )

        self._log(sql)
        schema = table.schema().to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, batcher(sql, schema=schema))

    def execute(
        self,
        expr: ir.Expr,
        limit: str | None = 'default',
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""
        import pandas as pd

        table = expr.as_table()
        sql = self.compile(table, limit=limit, **kwargs)

        schema = table.schema()
        self._log(sql)

        external_tables = self._collect_in_memory_tables(
            expr, external_tables=toolz.valmap(_to_memtable, external_tables or {})
        )
        external_data = self._normalize_external_tables(external_tables)
        df = self.con.query_df(
            sql, external_data=external_data, use_na_values=False, use_none=True
        )

        if df.empty:
            df = pd.DataFrame(columns=schema.names)

        result = PandasData.convert_table(df, schema)
        if isinstance(expr, ir.Scalar):
            return result.iat[0, 0]
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
        **kwargs,
    ) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        external_tables
            Mapping of table name to pandas DataFrames providing
            external datasources for the query
        kwargs
            Backend specific query arguments

        Returns
        -------
        Cursor
            Clickhouse cursor
        """
        external_tables = toolz.valmap(_to_memtable, external_tables or {})
        external_data = self._normalize_external_tables(external_tables)
        self._log(query)
        return self.con.query(query, external_data=external_data, **kwargs)

    def fetch_from_cursor(self, cursor, schema):
        import pandas as pd

        df = pd.DataFrame.from_records(iter(cursor), columns=schema.names)
        return PandasData.convert_table(df, schema)

    def close(self) -> None:
        """Close ClickHouse connection."""
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
        query = f"DESCRIBE {qualified_name}"
        with closing(self.raw_sql(query)) as results:
            names, types, *_ = results.result_columns
        return sch.Schema(dict(zip(names, map(parse, types))))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        query = f"EXPLAIN json = 1, description = 0, header = 1 {query}"
        with closing(self.raw_sql(query)) as results:
            [[raw_plans]] = results.result_columns
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
        with closing(
            self.raw_sql(f"CREATE DATABASE {if_not_exists}{name} ENGINE = {engine}")
        ):
            pass

    def drop_database(self, name: str, *, force: bool = False) -> None:
        if_exists = "IF EXISTS " * force
        with closing(self.raw_sql(f"DROP DATABASE {if_exists}{name}")):
            pass

    def truncate_table(self, name: str, database: str | None = None) -> None:
        ident = self._fully_qualified_name(name, database)
        with closing(self.raw_sql(f"TRUNCATE TABLE {ident}")):
            pass

    def drop_table(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        ident = self._fully_qualified_name(name, database)
        with closing(self.raw_sql(f"DROP TABLE {'IF EXISTS ' * force}{ident}")):
            pass

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        # backend specific arguments
        engine: str = "File(Native)",
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
            raise com.IbisError(
                "ClickHouse temporary tables are not yet supported due to a bug in `clickhouse_driver`"
            )

        tmp = "TEMPORARY " * temp
        replace = "OR REPLACE " * overwrite
        table = self._fully_qualified_name(name, database)
        code = f"CREATE {replace}{tmp}TABLE {table}"

        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")

        if obj is not None and not isinstance(obj, ir.Expr):
            obj = ibis.memtable(obj, schema=schema)

        if schema is None:
            schema = obj.schema()

        serialized_schema = ", ".join(
            f"`{name}` {serialize(typ)}" for name, typ in schema.items()
        )

        code += f" ({serialized_schema}) ENGINE = {engine}"

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
            code += f" AS {self.compile(obj)}"

        external_tables = self._collect_in_memory_tables(obj)

        # create the table
        self.con.raw_query(
            code, external_data=self._normalize_external_tables(external_tables)
        )

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
        with closing(
            self.raw_sql(code, external_tables=self._collect_in_memory_tables(obj))
        ):
            pass
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        name = self._fully_qualified_name(name, database)
        if_exists = "IF EXISTS " * force
        with closing(self.raw_sql(f"DROP VIEW {if_exists}{name}")):
            pass
