from __future__ import annotations

import contextlib
import functools
import sqlite3
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import UrlFromPath
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import C
from ibis.backends.sqlite.converter import SQLitePandasData
from ibis.backends.sqlite.udf import ignore_nulls, register_all

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow as pa


@functools.cache
def _init_sqlite3():
    import pandas as pd

    # required to support pandas Timestamp's from user input
    sqlite3.register_adapter(pd.Timestamp, pd.Timestamp.isoformat)


def _quote(name: str) -> str:
    return sg.to_identifier(name, quoted=True).sql("sqlite")


class Backend(SQLBackend, UrlFromPath):
    name = "sqlite"
    compiler = sc.sqlite.compiler
    supports_python_udfs = True

    @property
    def current_database(self) -> str:
        return "main"

    @property
    def version(self) -> str:
        return sqlite3.sqlite_version

    def do_connect(
        self,
        database: str | Path | None = None,
        type_map: dict[str, str | dt.DataType] | None = None,
    ) -> None:
        """Create an Ibis client connected to a SQLite database.

        Multiple database files can be accessed using the `attach()` method.

        Parameters
        ----------
        database
            File path to the SQLite database file. If `None`, creates an
            in-memory transient database and you can use attach() to add more
            files
        type_map
            An optional mapping from a string name of a SQLite "type" to the
            corresponding Ibis DataType that it represents. This can be used
            to override schema inference for a given SQLite database.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.sqlite.connect()
        >>> t = con.create_table("my_table", schema=ibis.schema(dict(x="int64")))
        >>> con.insert("my_table", obj=[(1,), (2,), (3,)])
        >>> t
        DatabaseTable: my_table
          x int64
        >>> t.head(1).execute()
           x
        0  1
        """
        _init_sqlite3()

        self.con = sqlite3.connect(":memory:" if database is None else database)

        self._post_connect(type_map)

    @util.experimental
    @classmethod
    def from_connection(
        cls,
        con: sqlite3.Connection,
        type_map: dict[str, str | dt.DataType] | None = None,
    ) -> Backend:
        """Create an Ibis client from an existing connection to a SQLite database.

        Parameters
        ----------
        con
            An existing connection to a SQLite database.
        type_map
            An optional mapping from a string name of a SQLite "type" to the
            corresponding Ibis DataType that it represents. This can be used
            to override schema inference for a given SQLite database.
        """
        new_backend = cls(type_map=type_map)
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect(type_map)
        return new_backend

    def _post_connect(
        self, type_map: dict[str, str | dt.DataType] | None = None
    ) -> None:
        if type_map:
            self._type_map = {k.lower(): ibis.dtype(v) for k, v in type_map.items()}
        else:
            self._type_map = {}

        register_all(self.con)
        self.con.execute("PRAGMA case_sensitive_like=ON")

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        if not isinstance(query, str):
            query = query.sql(dialect=self.name)
        return self.con.execute(query, **kwargs)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    @contextlib.contextmanager
    def begin(self):
        cur = self.con.cursor()
        try:
            yield cur
        except Exception:
            self.con.rollback()
            raise
        else:
            self.con.commit()
        finally:
            cur.close()

    def list_databases(self, like: str | None = None) -> list[str]:
        with self._safe_raw_sql("SELECT name FROM pragma_database_list()") as cur:
            results = [r[0] for r in cur.fetchall()]

        return sorted(self._filter_with_like(results, like))

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List the tables in the database.

        If `database` is None, the current database is used, and temporary
        tables are included in the result.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database.
        """
        if database is None:
            database = "main"
            schemas = [database, "temp"]
        else:
            schemas = [database]

        sql = (
            sg.select(C.name)
            .from_(sg.func("pragma_table_list"))
            .where(
                C.schema.isin(*map(sge.convert, schemas)),
                C.type.isin(sge.convert("table"), sge.convert("view")),
                ~C.name.isin(
                    sge.convert("sqlite_schema"),
                    sge.convert("sqlite_master"),
                    sge.convert("sqlite_temp_schema"),
                    sge.convert("sqlite_temp_master"),
                ),
            )
            .sql(self.dialect)
        )
        with self._safe_raw_sql(sql) as cur:
            results = [r[0] for r in cur.fetchall()]

        return sorted(self._filter_with_like(results, like))

    def _parse_type(self, typ: str, nullable: bool) -> dt.DataType:
        typ = typ.lower()
        try:
            out = self._type_map[typ]
        except KeyError:
            return self.compiler.type_mapper.from_string(typ, nullable=nullable)
        else:
            return out.copy(nullable=nullable)

    def _inspect_schema(
        self, cur: sqlite3.Cursor, table_name: str, database: str | None = None
    ) -> sch.Schema:
        if database is None:
            database = "main"

        quoted = self.compiler.quoted
        quoted_db = sg.to_identifier(database, quoted=quoted)
        quoted_table = sg.to_identifier(table_name, quoted=quoted)

        sql = (
            sg.select("name", "type", sg.to_identifier("notnull", quoted=quoted))
            .from_(
                sge.Table(
                    this=self.compiler.f.anon.pragma_table_info(quoted_table),
                    db=quoted_db,
                )
            )
            .sql(self.dialect)
        )
        cur.execute(sql)
        rows = cur.fetchall()
        if not rows:
            raise com.TableNotFound(table_name)

        table_info = {name: (typ, not notnull) for name, typ, notnull in rows}

        # if no type info was returned for a column, fetch the type of the
        # first row and assume that matches the rest of the rows
        unknown = [name for name, (typ, _) in table_info.items() if not typ]
        if unknown:
            queries = (
                self.compiler.f.typeof(sg.to_identifier(name, quoted=quoted))
                for name in unknown
            )
            cur.execute(
                sg.select(*queries)
                .from_(sg.table(table_name, db=database, quoted=quoted))
                .limit(1)
                .sql(self.dialect)
            )
            row = cur.fetchone()
            if row is not None:
                for name, typ in zip(unknown, row):
                    _, nullable = table_info[name]
                    table_info[name] = (typ, nullable)
            else:
                raise com.IbisError(f"Failed to infer types for columns {unknown}")

        return sch.Schema(
            {
                name: self._parse_type(typ, nullable)
                for name, (typ, nullable) in table_info.items()
            }
        )

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Compute the schema of a `table`.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        catalog
            Catalog name. Unused for sqlite.
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema

        """
        if catalog is not None:
            raise TypeError("sqlite doesn't support `catalog`, use `database` instead")
        with self.begin() as cur:
            return self._inspect_schema(cur, table_name, database)

    def _get_schema_using_query(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self.begin() as cur:
            # create a view that should only be visible in this transaction
            view = util.gen_name("ibis_sqlite_metadata")
            cur.execute(f"CREATE TEMPORARY VIEW {view} AS {query}")

            try:
                return self._inspect_schema(cur, view, database="temp")
            finally:
                # drop the view when we're done with it
                cur.execute(f"DROP VIEW IF EXISTS {view}")

    def _fetch_from_cursor(
        self, cursor: sqlite3.Cursor, schema: sch.Schema
    ) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame.from_records(cursor, columns=schema.names, coerce_float=True)
        return SQLitePandasData.convert_table(df, schema)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        import pyarrow as pa

        self._run_pre_execute_hooks(expr)

        schema = expr.as_table().schema()
        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            df = self._fetch_from_cursor(cursor, schema)
        table = pa.Table.from_pandas(
            df, schema=schema.to_pyarrow(), preserve_index=False
        )
        return table.to_reader(max_chunksize=chunk_size)

    def _generate_create_table(self, table: sge.Table, schema: sch.Schema):
        target = sge.Schema(this=table, expressions=schema.to_sqlglot(self.dialect))

        return sge.Create(kind="TABLE", this=target)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        table = sg.table(op.name, quoted=self.compiler.quoted, catalog="temp")
        create_stmt = self._generate_create_table(table, op.schema).sql(self.name)
        df = op.data.to_frame()

        data = df.itertuples(index=False)
        insert_stmt = self._build_insert_template(
            op.name, schema=op.schema, catalog="temp", columns=True
        )

        with self.begin() as cur:
            cur.execute(create_stmt)
            cur.executemany(insert_stmt, data)

    def _register_udfs(self, expr: ir.Expr) -> None:
        import ibis.expr.operations as ops

        con = self.con

        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_register_{udf_node.__input_type__.name.lower()}_udf"
            )
            registration_func = compile_func(udf_node)
            if registration_func is not None:
                registration_func(con)

    def _register_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        name = type(udf_node).__name__
        nargs = len(udf_node.__signature__.parameters)
        func = udf_node.__func__

        def check_dtype(dtype, name=None):
            if not (
                dtype.is_string()
                or dtype.is_binary()
                or dtype.is_numeric()
                or dtype.is_boolean()
            ):
                label = "return value" if name is None else f"argument `{name}`"
                raise com.IbisTypeError(
                    "SQLite only supports strings, bytes, booleans and numbers as UDF input and output, "
                    f"{label} has unsupported type {dtype}"
                )

        for argname, arg in zip(udf_node.argnames, udf_node.args):
            check_dtype(arg.dtype, argname)
        check_dtype(udf_node.dtype)

        def register_udf(con):
            return con.create_function(name, nargs, ignore_nulls(func))

        return register_udf

    def attach(self, name: str, path: str | Path) -> None:
        """Connect another SQLite database file to the current connection.

        Parameters
        ----------
        name
            Database name within SQLite
        path
            Path to sqlite3 database files

        Examples
        --------
        >>> con1 = ibis.sqlite.connect("/tmp/original.db")
        >>> con2 = ibis.sqlite.connect("/tmp/new.db")
        >>> con1.attach("new", "/tmp/new.db")
        >>> con1.list_tables(database="new")
        []
        """
        with self.begin() as cur:
            cur.execute(f"ATTACH DATABASE {str(path)!r} AS {_quote(name)}")

    def create_table(
        self,
        name: str,
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
    ):
        """Create a table in SQLite.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists

        """
        if schema is None and obj is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if schema is not None:
            schema = ibis.schema(schema)

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                obj = ibis.memtable(obj)

            self._run_pre_execute_hooks(obj)

            insert_query = self.compiler.to_sqlglot(obj)
        else:
            insert_query = None

        if temp:
            if database not in (None, "temp"):
                raise ValueError(
                    "SQLite doesn't support creating temporary tables in an explicit database"
                )
            else:
                database = "temp"

        if overwrite:
            created_table = sg.table(
                util.gen_name(f"{self.name}_table"),
                catalog=database,
                quoted=self.compiler.quoted,
            )
            table = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        else:
            created_table = table = sg.table(
                name, catalog=database, quoted=self.compiler.quoted
            )

        create_stmt = self._generate_create_table(
            created_table, schema=(schema or obj.schema())
        ).sql(self.name)

        with self.begin() as cur:
            cur.execute(create_stmt)

            if insert_query is not None:
                cur.execute(
                    sge.Insert(this=created_table, expression=insert_query).sql(
                        self.name
                    )
                )

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=table, exists=True).sql(self.name)
                )
                # SQLite's ALTER TABLE statement doesn't support using a
                # fully-qualified table reference after RENAME TO. Since we
                # never rename between databases, we only need the table name
                # here.
                quoted_name = _quote(name)
                cur.execute(
                    f"ALTER TABLE {created_table.sql(self.name)} RENAME TO {quoted_name}"
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def drop_table(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        drop_stmt = sg.exp.Drop(
            kind="TABLE",
            this=sg.table(name, catalog=database, quoted=self.compiler.quoted),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        view = sg.table(name, catalog=database, quoted=self.compiler.quoted)

        stmts = []
        if overwrite:
            stmts.append(sge.Drop(kind="VIEW", this=view, exists=True).sql(self.name))
        stmts.append(
            sge.Create(
                this=view, kind="VIEW", replace=False, expression=self.compile(obj)
            ).sql(self.name)
        )

        self._run_pre_execute_hooks(obj)

        with self.begin() as cur:
            for stmt in stmts:
                cur.execute(stmt)

        return self.table(name, database=database)

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.
        overwrite
            If `True` then replace existing contents of table

        Raises
        ------
        NotImplementedError
            If inserting data from a different database
        ValueError
            If the type of `obj` isn't supported

        """
        table = sg.table(table_name, catalog=database, quoted=self.compiler.quoted)
        if not isinstance(obj, ir.Expr):
            obj = ibis.memtable(obj)

        self._run_pre_execute_hooks(obj)

        query = self._build_insert_from_table(
            target=table_name, source=obj, catalog=database
        )
        insert_stmt = query.sql(self.name)

        with self.begin() as cur:
            if overwrite:
                cur.execute(sge.Delete(this=table).sql(self.dialect))
            cur.execute(insert_stmt)
