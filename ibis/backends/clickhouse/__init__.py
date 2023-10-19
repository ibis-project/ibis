from __future__ import annotations

import ast
import atexit
import glob
import json
from contextlib import closing, suppress
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import clickhouse_connect as cc
import pyarrow as pa
import sqlalchemy as sa
import sqlglot as sg
import toolz
from clickhouse_connect.driver.external import ExternalData

import ibis
import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import BaseBackend, CanCreateDatabase
from ibis.backends.base.sqlglot import STAR, C, F
from ibis.backends.clickhouse.compiler import translate
from ibis.backends.clickhouse.datatypes import ClickhouseType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

    import pandas as pd

    from ibis.common.typing import SupportsSchema


def _to_memtable(v):
    return ibis.memtable(v).op() if not isinstance(v, ops.InMemoryTable) else v


class Backend(BaseBackend, CanCreateDatabase):
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
        self._temp_views = set()

    @property
    def version(self) -> str:
        return self.con.server_version

    @property
    def current_database(self) -> str:
        with closing(self.raw_sql(sg.select(F.currentDatabase()))) as result:
            [(db,)] = result.result_rows
        return db

    def list_databases(self, like: str | None = None) -> list[str]:
        with closing(
            self.raw_sql(sg.select(C.name).from_(sg.table("databases", db="system")))
        ) as result:
            results = result.result_columns

        if results:
            (databases,) = results
        else:
            databases = []
        return self._filter_with_like(databases, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = sg.select(C.name).from_(sg.table("tables", db="system"))

        if database is None:
            database = F.currentDatabase()
        else:
            database = sg.exp.convert(database)

        query = query.where(C.database.eq(database).or_(C.is_temporary))

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
                raise TypeError(f"Schema is empty for external table {name}")

            structure = [
                f"{name} {ClickhouseType.to_string(typ.copy(nullable=not typ.is_nested()))}"
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
        self, expr: ir.Table | None, external_tables: Mapping | None = None
    ):
        memtables = {op.name: op for op in expr.op().find(ops.InMemoryTable)}
        externals = toolz.valmap(_to_memtable, external_tables or {})
        return toolz.merge(memtables, externals)

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
            table = reader.read_all()

        return expr.__pyarrow_result__(table)

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

        external_tables = self._collect_in_memory_tables(expr, external_tables)
        external_data = self._normalize_external_tables(external_tables)

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
        limit: str | None = "default",
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""
        import pandas as pd

        table = expr.as_table()
        sql = self.compile(table, limit=limit, **kwargs)

        schema = table.schema()
        self._log(sql)

        external_tables = self._collect_in_memory_tables(expr, external_tables)
        external_data = self._normalize_external_tables(external_tables)
        df = self.con.query_df(
            sql, external_data=external_data, use_na_values=False, use_none=True
        )

        if df.empty:
            df = pd.DataFrame(columns=schema.names)

        # TODO: remove the extra conversion
        #
        # the extra __pandas_result__ call is to work around slight differences
        # in single column conversion and whole table conversion
        return expr.__pandas_result__(table.__pandas_result__(df))

    def _to_sqlglot(
        self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
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
            sql = sg.select(STAR).from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a ClickHouse SQL string."""
        return self._to_sqlglot(expr, limit=limit, params=params, **kwargs).sql(
            dialect=self.name, pretty=True
        )

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        return self.compile(expr, **kwargs)

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
        op = ops.DatabaseTable(
            name=name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(database=database),
        )
        return op.to_expr()

    def insert(
        self,
        name: str,
        obj: pd.DataFrame,
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        import pandas as pd
        import pyarrow as pa

        if not isinstance(obj, pd.DataFrame):
            raise com.IbisError(
                f"Invalid input type {type(obj)}; only pandas DataFrames are accepted as input"
            )

        # TODO(cpcloud): add support for arrow tables
        # TODO(cpcloud): insert_df doesn't work with pandas 2.1.0, move back to
        # that (maybe?) when `clickhouse_connect` is fixed
        t = pa.Table.from_pandas(obj)
        return self.con.insert_arrow(name, t, settings=settings, **kwargs)

    def raw_sql(
        self,
        query: str | sg.exp.Expression,
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
        with suppress(AttributeError):
            query = query.sql(dialect=self.name, pretty=True)
        self._log(query)
        return self.con.query(query, external_data=external_data, **kwargs)

    def close(self) -> None:
        """Close ClickHouse connection."""
        self.con.close()

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
        query = sg.exp.Describe(this=sg.table(table_name, db=database))
        with closing(self.raw_sql(query)) as results:
            names, types, *_ = results.result_columns
        return sch.Schema(dict(zip(names, map(ClickhouseType.from_string, types))))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        query = f"EXPLAIN json = 1, description = 0, header = 1 {query}"
        with closing(self.raw_sql(query)) as results:
            [[raw_plans]] = results.result_columns
        [plan] = json.loads(raw_plans)
        return sch.Schema(
            {
                field["Name"]: ClickhouseType.from_string(field["Type"])
                for field in plan["Plan"]["Header"]
            }
        )

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        from ibis.backends.clickhouse.compiler.values import translate_val

        return translate_val.dispatch(operation) is not translate_val.dispatch(object)

    def create_database(
        self, name: str, *, force: bool = False, engine: str = "Atomic"
    ) -> None:
        src = sg.exp.Create(
            this=sg.to_identifier(name),
            kind="DATABASE",
            exists=force,
            properties=sg.exp.Properties(
                expressions=[sg.exp.EngineProperty(this=sg.to_identifier(engine))]
            ),
        )
        with closing(self.raw_sql(src)):
            pass

    def drop_database(self, name: str, *, force: bool = False) -> None:
        src = sg.exp.Drop(this=sg.to_identifier(name), kind="DATABASE", exists=force)
        with closing(self.raw_sql(src)):
            pass

    def truncate_table(self, name: str, database: str | None = None) -> None:
        ident = sg.table(name, db=database).sql(self.name)
        with closing(self.raw_sql(f"TRUNCATE TABLE {ident}")):
            pass

    def drop_table(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        src = sg.exp.Drop(this=sg.table(name, db=database), kind="TABLE", exists=force)
        with closing(self.raw_sql(src)):
            pass

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        engine: str = "MergeTree",
        **kwargs: Any,
    ) -> ir.Table:
        import pyarrow.dataset as ds
        from clickhouse_connect.driver.tools import insert_file

        from ibis.formats.pyarrow import PyArrowSchema

        paths = list(glob.glob(str(path)))
        schema = PyArrowSchema.to_ibis(ds.dataset(paths, format="parquet").schema)

        name = table_name or util.gen_name("read_parquet")
        table = self.create_table(name, engine=engine, schema=schema, temp=True)

        for file_path in paths:
            insert_file(
                client=self.con,
                table=name,
                file_path=file_path,
                fmt="Parquet",
                **kwargs,
            )
        return table

    def read_csv(
        self,
        path: str | Path,
        table_name: str | None = None,
        engine: str = "MergeTree",
        **kwargs: Any,
    ) -> ir.Table:
        import pyarrow.dataset as ds
        from clickhouse_connect.driver.tools import insert_file

        from ibis.formats.pyarrow import PyArrowSchema

        paths = list(glob.glob(str(path)))
        schema = PyArrowSchema.to_ibis(ds.dataset(paths, format="csv").schema)

        name = table_name or util.gen_name("read_csv")
        table = self.create_table(name, engine=engine, schema=schema, temp=True)

        for file_path in paths:
            insert_file(client=self.con, table=name, file_path=file_path, **kwargs)
        return table

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
        engine: str = "MergeTree",
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
            for specifics. Defaults to [`MergeTree`](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree)
            with `ORDER BY tuple()` because `MergeTree` is the most
            feature-complete engine.
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
        if temp and overwrite:
            raise com.IbisInputError(
                "Cannot specify both `temp=True` and `overwrite=True` for ClickHouse"
            )

        if obj is None and schema is None:
            raise com.IbisError("The `schema` or `obj` parameter is required")

        if obj is not None and not isinstance(obj, ir.Expr):
            obj = ibis.memtable(obj, schema=schema)

        if schema is None:
            schema = obj.schema()

        this = sg.exp.Schema(
            this=sg.table(name, db=database),
            expressions=[
                sg.exp.ColumnDef(
                    this=sg.to_identifier(name), kind=ClickhouseType.from_ibis(typ)
                )
                for name, typ in schema.items()
            ],
        )
        properties = [
            # the engine cannot be quoted, since clickhouse won't allow e.g.,
            # "File(Native)"
            sg.exp.EngineProperty(this=sg.to_identifier(engine, quoted=False))
        ]

        if temp:
            properties.append(sg.exp.TemporaryProperty())

        if order_by is not None or engine == "MergeTree":
            # engine == "MergeTree" requires an order by clause, which is the
            # empty tuple if order_by is False-y
            properties.append(
                sg.exp.Order(
                    expressions=[
                        sg.exp.Ordered(
                            this=sg.exp.Tuple(
                                expressions=list(map(sg.column, order_by or ()))
                            )
                        )
                    ]
                )
            )

        if partition_by is not None:
            properties.append(
                sg.exp.PartitionedByProperty(
                    this=sg.exp.Schema(
                        expressions=list(map(sg.to_identifier, partition_by))
                    )
                )
            )

        if sample_by is not None:
            properties.append(
                sg.exp.SampleProperty(
                    this=sg.exp.Tuple(expressions=list(map(sg.column, sample_by)))
                )
            )

        if settings:
            properties.append(
                sg.exp.SettingsProperty(
                    expressions=[
                        sg.exp.SetItem(
                            this=sg.exp.EQ(
                                this=sg.to_identifier(name),
                                expression=sg.exp.convert(value),
                            )
                        )
                        for name, value in settings.items()
                    ]
                )
            )

        external_tables = {}
        expression = None

        if obj is not None:
            expression = self._to_sqlglot(obj)
            external_tables.update(self._collect_in_memory_tables(obj))

        code = sg.exp.Create(
            this=this,
            kind="TABLE",
            replace=overwrite,
            expression=expression,
            properties=sg.exp.Properties(expressions=properties),
        )

        external_data = self._normalize_external_tables(external_tables)

        # create the table
        sql = code.sql(self.name, pretty=True)
        self.con.raw_query(sql, external_data=external_data)

        return self.table(name, database=database)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        src = sg.exp.Create(
            this=sg.table(name, db=database),
            kind="VIEW",
            replace=overwrite,
            expression=self._to_sqlglot(obj),
        )
        external_tables = self._collect_in_memory_tables(obj)
        with closing(self.raw_sql(src, external_tables=external_tables)):
            pass
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        src = sg.exp.Drop(this=sg.table(name, db=database), kind="VIEW", exists=force)
        with closing(self.raw_sql(src)):
            pass

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(op.name)

    def _create_temp_view(self, table_name, source):
        if table_name not in self._temp_views and table_name in self.list_tables():
            raise ValueError(
                f"{table_name} already exists as a non-temporary table or view"
            )
        src = sg.exp.Create(
            this=sg.table(table_name), kind="VIEW", replace=True, expression=source
        )
        self.raw_sql(src)
        self._temp_views.add(table_name)
        self._register_temp_view_cleanup(table_name)

    def _register_temp_view_cleanup(self, name: str) -> None:
        def drop(self, name: str, query: str):
            self.raw_sql(query)
            self._temp_views.discard(name)

        query = sg.exp.Drop(this=sg.table(name), kind="VIEW", exists=True)
        atexit.register(drop, self, name=name, query=query)
