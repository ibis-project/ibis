from __future__ import annotations

import contextlib
import functools
import sqlite3
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.base.sqlglot.compiler import C, F
from ibis.backends.sqlite.compiler import SQLiteCompiler
from ibis.backends.sqlite.udf import register_all
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa


@functools.cache
def _init_sqlite3():
    import pandas as pd

    # TODO: can we remove this?
    sqlite3.register_adapter(pd.Timestamp, lambda value: value.isoformat())


class Backend(SQLGlotBackend):
    name = "sqlite"
    compiler = SQLiteCompiler()

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
            corresponding ibis DataType that it represents. This can be used
            to override schema inference for a given SQLite database.

        Examples
        --------
        >>> import ibis
        >>> ibis.sqlite.connect("path/to/my/sqlite.db")
        """
        _init_sqlite3()

        if type_map:
            self._type_map = {k.lower(): v for k, v in type_map.items()}
        else:
            self._type_map = {}

        self.con = sqlite3.connect(":memory:" if database is None else database)

        register_all(self.con)
        self.con.execute("PRAGMA case_sensitive_like=ON")

    def _from_url(self, url: str, **kwargs):
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
        url = urlparse(url)
        database = url.path[1:] or ":memory:"
        return self.connect(database=database, **kwargs)

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
        self,
        like: str | None = None,
        database: str | None = None,
    ) -> list[str]:
        if database is None:
            database = "main"

        sql = (
            sg.select("name")
            .from_(F.pragma_table_list())
            .where(
                C.schema.eq(database),
                C.type.isin("table", "view"),
                ~(
                    C.name.isin(
                        "sqlite_schema",
                        "sqlite_master",
                        "sqlite_temp_schema",
                        "sqlite_temp_master",
                    )
                ),
            )
            .sql(self.name)
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
    ) -> Iterator[tuple[str, dt.DataType]]:
        if database is None:
            database = "main"

        sql = f'SELECT name, type, "notnull" FROM "{database}".pragma_table_info("{table_name}")'
        cur.execute(sql)
        rows = cur.fetchall()
        if not rows:
            raise com.IbisError(f"Table not found: {table_name!r}")

        table_info = {name: (typ, not notnull) for name, typ, notnull in rows}

        # if no type info was returned for a column, fetch the type of the
        # first row and assume that matches the rest of the rows
        unknown = [name for name, (typ, _) in table_info.items() if not typ]
        if unknown:
            queries = ", ".join(f'typeof("{name}")' for name in unknown)
            cur.execute(f'SELECT {queries} FROM "{database}"."{table_name}" LIMIT 1')
            row = cur.fetchone()
            if row is not None:
                for name, (typ,) in zip(unknown, row):
                    _, nullable = table_info[name]
                    table_info[name] = (typ, nullable)
            else:
                raise com.IbisError(f"Failed to infer types for columns {unknown}")

        for name, (typ, nullable) in table_info.items():
            yield name, self._parse_type(typ, nullable)

    def get_schema(
        self, table_name: str, schema: str | None = None, database: str | None = None
    ) -> sch.Schema:
        """Compute the schema of a `table`.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        schema
            Schema name. Unused for sqlite.
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        if schema is not None:
            raise TypeError("sqlite doesn't support `schema`, use `database` instead")
        with self.begin() as cur:
            return sch.Schema.from_tuples(
                self._inspect_schema(cur, table_name, database)
            )

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self.begin() as cur:
            # create a view that should only be visible in this transaction
            view = f"__ibis_sqlite_metadata_{util.guid()}"
            cur.execute(f"CREATE TEMPORARY VIEW {view} AS {query}")

            yield from self._inspect_schema(cur, view)

            # drop the view when we're done with it
            cur.execute(f"DROP VIEW IF EXISTS {view}")

    def _fetch_from_cursor(
        self, cursor: sqlite3.Cursor, schema: sch.Schema
    ) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame.from_records(cursor, columns=schema.names, coerce_float=True)
        return PandasData.convert_table(df, schema)

    def _generate_create_table(self, table: sge.Table, schema: sch.Schema):
        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=self.compiler.quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for colname, typ in schema.items()
        ]

        target = sge.Schema(this=table, expressions=column_defs)

        return sge.Create(kind="TABLE", this=target)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        # only register if we haven't already done so
        if op.name not in self.list_tables():
            table = sg.table(op.name, quoted=self.compiler.quoted, catalog="temp")
            create_stmt = self._generate_create_table(table, op.schema).sql(self.name)
            df = op.data.to_frame()

            data = df.itertuples(index=False)
            cols = ", ".join(
                sg.to_identifier(col, quoted=self.compiler.quoted).sql(self.name)
                for col in op.schema.keys()
            )
            specs = ", ".join(["?"] * len(op.schema))
            insert_stmt = (
                f"INSERT INTO {table.sql(self.name)} ({cols}) VALUES ({specs})"
            )

            with self.begin() as cur:
                cur.execute(create_stmt)
                cur.executemany(insert_stmt, data)

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
        >>> con1 = ibis.sqlite.connect("original.db")
        >>> con2 = ibis.sqlite.connect("new.db")
        >>> con1.attach("new", "new.db")
        >>> con1.list_tables(database="new")
        """
        with self.begin() as cur:
            cur.execute(f'ATTACH DATABASE {str(path)!r} AS "{name}"')

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
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

            insert_query = self._to_sqlglot(obj)
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
                cur.execute(
                    f"ALTER TABLE {created_table.sql(self.name)} RENAME TO {table.sql(self.name)}"
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
