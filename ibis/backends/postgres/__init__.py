"""PostgreSQL backend."""

from __future__ import annotations

import contextlib
import inspect
import textwrap
from functools import partial
from itertools import repeat, takewhile
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import parse_qs, urlparse

import psycopg2
import sqlglot as sg
import sqlglot.expressions as sge
from psycopg2 import extras

import ibis
import ibis.common.exceptions as com
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.base.sqlglot.compiler import TRUE, C, ColGen, F
from ibis.backends.postgres.compiler import PostgresCompiler
from ibis.common.exceptions import InvalidDecoratorError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import pyarrow as pa


def _verify_source_line(func_name: str, line: str):
    if line.startswith("@"):
        raise InvalidDecoratorError(func_name, line)
    return line


class Backend(SQLGlotBackend):
    name = "postgres"
    compiler = PostgresCompiler()
    supports_python_udfs = True

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
        database, *schema = url.path[1:].split("/", 1)
        query_params = parse_qs(url.query)
        connect_args = {
            "user": url.username,
            "password": url.password or "",
            "host": url.hostname,
            "database": database or "",
            "schema": schema[0] if schema else "",
        }

        for name, value in query_params.items():
            if len(value) > 1:
                connect_args[name] = value
            elif len(value) == 1:
                connect_args[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        kwargs.update(connect_args)
        self._convert_kwargs(kwargs)

        if "user" in kwargs and not kwargs["user"]:
            del kwargs["user"]

        if "host" in kwargs and not kwargs["host"]:
            del kwargs["host"]

        if "database" in kwargs and not kwargs["database"]:
            del kwargs["database"]

        if "schema" in kwargs and not kwargs["schema"]:
            del kwargs["schema"]

        if "password" in kwargs and kwargs["password"] is None:
            del kwargs["password"]

        return self.connect(**kwargs)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [
                            sg.exp.ColumnConstraint(
                                kind=sg.exp.NotNullColumnConstraint()
                            )
                        ]
                    ),
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
                properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
            )
            create_stmt_sql = create_stmt.sql(self.dialect)

            columns = schema.keys()
            df = op.data.to_frame()
            data = df.itertuples(index=False)
            cols = ", ".join(
                ident.sql(self.dialect)
                for ident in map(partial(sg.to_identifier, quoted=quoted), columns)
            )
            specs = ", ".join(repeat("%s", len(columns)))
            table = sg.table(name, quoted=quoted)
            sql = f"INSERT INTO {table.sql(self.dialect)} ({cols}) VALUES ({specs})"
            with self.begin() as cur:
                cur.execute(create_stmt_sql)
                extras.execute_batch(cur, sql, data, 128)

    @contextlib.contextmanager
    def begin(self):
        con = self.con
        cursor = con.cursor()
        try:
            yield cursor
        except Exception:
            con.rollback()
            raise
        else:
            con.commit()
        finally:
            cursor.close()

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.backends.postgres.converter import PostgresPandasData

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = PostgresPandasData.convert_table(df, schema)
        return df

    @property
    def version(self):
        version = f"{self.con.server_version:0>6}"
        major = int(version[:2])
        minor = int(version[2:4])
        patch = int(version[4:])
        pieces = [major]
        if minor:
            pieces.append(minor)
        pieces.append(patch)
        return ".".join(map(str, pieces))

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to PostgreSQL database.

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
        schema
            PostgreSQL schema to use. If `None`, use the default `search_path`.
        kwargs
            Additional keyword arguments to pass to the backend client connection.

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_POSTGRES_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_POSTGRES_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_POSTGRES_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_POSTGRES_DATABASE", "ibis_testing")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        PostgreSQLTable[table]
          name: functional_alltypes
          schema:
            id : int32
            bool_col : boolean
            tinyint_col : int16
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32

        """

        self.con = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            options=(f"-csearch_path={schema}" * (schema is not None)) or None,
            **kwargs,
        )

        with self.begin() as cur:
            cur.execute("SET TIMEZONE = UTC")

    def list_tables(
        self, like: str | None = None, schema: str | None = None
    ) -> list[str]:
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        schema
            The schema to perform the list against.

            ::: {.callout-warning}
            ## `schema` refers to database hierarchy

            The `schema` parameter does **not** refer to the column names and
            types of `table`.
            :::

        """
        conditions = [TRUE]

        if schema is not None:
            conditions = C.table_schema.eq(sge.convert(schema))

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql(self.dialect)
        )

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def list_databases(self, like=None) -> list[str]:
        # http://dba.stackexchange.com/a/1304/58517
        dbs = (
            sg.select(C.datname)
            .from_(sg.table("pg_database", db="pg_catalog"))
            .where(sg.not_(C.datistemplate))
        )
        with self._safe_raw_sql(dbs) as cur:
            databases = list(map(itemgetter(0), cur))

        return self._filter_with_like(databases, like)

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(F.current_database())) as cur:
            (db,) = cur.fetchone()
        return db

    @property
    def current_schema(self) -> str:
        with self._safe_raw_sql(sg.select(F.current_schema())) as cur:
            (schema,) = cur.fetchone()
        return schema

    def function(self, name: str, *, schema: str | None = None) -> Callable:
        n = ColGen(table="n")
        p = ColGen(table="p")
        f = self.compiler.f

        predicates = [p.proname.eq(name)]

        if schema is not None:
            predicates.append(n.nspname.rlike(sge.convert(f"^({schema})$")))

        query = (
            sg.select(
                f["pg_catalog.pg_get_function_result"](p.oid).as_("return_type"),
                f.string_to_array(
                    f["pg_catalog.pg_get_function_arguments"](p.oid), ", "
                ).as_("signature"),
            )
            .from_(sg.table("pg_proc", db="pg_catalog").as_("p"))
            .join(
                sg.table("pg_namespace", db="pg_catalog").as_("n"),
                on=n.oid.eq(p.pronamespace),
                join_type="LEFT",
            )
            .where(sg.and_(*predicates))
        )

        def split_name_type(arg: str) -> tuple[str, dt.DataType]:
            name, typ = arg.split(" ", 1)
            return name, self.compiler.type_mapper.from_string(typ)

        with self._safe_raw_sql(query) as cur:
            rows = cur.fetchall()

        if not rows:
            name = f"{schema}.{name}" if schema else name
            raise exc.MissingUDFError(name)
        elif len(rows) > 1:
            raise exc.AmbiguousUDFError(name)

        [(raw_return_type, signature)] = rows
        return_type = self.compiler.type_mapper.from_string(raw_return_type)
        signature = list(map(split_name_type, signature))

        # dummy callable
        def fake_func(*args, **kwargs):
            ...

        fake_func.__name__ = name
        fake_func.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ
                )
                for name, typ in signature
            ],
            return_annotation=return_type,
        )
        fake_func.__annotations__ = {"return": return_type, **dict(signature)}
        op = ops.udf.scalar.builtin(fake_func, schema=schema)
        return op

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        config = udf_node.__config__
        func = udf_node.__func__
        func_name = func.__name__

        lines, _ = inspect.getsourcelines(func)
        iter_lines = iter(lines)

        function_premable_lines = list(
            takewhile(lambda line: not line.lstrip().startswith("def "), iter_lines)
        )

        if len(function_premable_lines) > 1:
            raise InvalidDecoratorError(
                name=func_name, lines="".join(function_premable_lines)
            )

        source = textwrap.dedent(
            "".join(map(partial(_verify_source_line, func_name), iter_lines))
        ).strip()

        type_mapper = self.compiler.type_mapper
        argnames = udf_node.argnames
        return dict(
            name=type(udf_node).__name__,
            ident=self.compiler.__sql_name__(udf_node),
            signature=", ".join(
                f"{argname} {type_mapper.to_string(arg.dtype)}"
                for argname, arg in zip(argnames, udf_node.args)
            ),
            return_type=type_mapper.to_string(udf_node.dtype),
            language=config.get("language", "plpython3u"),
            source=source,
            args=", ".join(argnames),
        )

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:
        """No op."""

    def _compile_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(f"pyarrow UDFs are not supported in {self.name}")

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(f"pandas UDFs are not supported in {self.name}")

    def _define_udf_translation_rules(self, expr: ir.Expr) -> None:
        """No-op, these are defined in the compiler."""

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
CREATE OR REPLACE FUNCTION {ident}({signature})
RETURNS {return_type}
LANGUAGE {language}
AS $$
{source}
return {name}({args})
$$""".format(**self._get_udf_source(udf_node))

    def _register_udfs(self, expr: ir.Expr) -> None:
        udf_sources = []
        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            if sql := compile_func(udf_node):
                udf_sources.append(sql)
        if udf_sources:
            # define every udf in one execution to avoid the overhead of
            # database round trips per udf
            with self._safe_raw_sql(";\n".join(udf_sources)):
                pass

    def get_schema(
        self, name: str, schema: str | None = None, database: str | None = None
    ):
        a = ColGen(table="a")
        c = ColGen(table="c")
        n = ColGen(table="n")

        format_type = self.compiler.f["pg_catalog.format_type"]

        type_info = (
            sg.select(
                a.attname.as_("column_name"),
                format_type(a.atttypid, a.atttypmod).as_("data_type"),
                sg.not_(a.attnotnull).as_("nullable"),
            )
            .from_(sg.table("pg_attribute", db="pg_catalog").as_("a"))
            .join(
                sg.table("pg_class", db="pg_catalog").as_("c"),
                on=c.oid.eq(a.attrelid),
                join_type="INNER",
            )
            .join(
                sg.table("pg_namespace", db="pg_catalog").as_("n"),
                on=n.oid.eq(c.relnamespace),
                join_type="INNER",
            )
            .where(
                a.attnum > 0,
                sg.not_(a.attisdropped),
                n.nspname.eq(schema) if schema is not None else TRUE,
                c.relname.eq(name),
            )
            .order_by(a.attnum)
        )

        type_mapper = self.compiler.type_mapper

        with self._safe_raw_sql(type_info) as cur:
            rows = cur.fetchall()

        if not rows:
            raise com.IbisError(f"Table not found: {name!r}")

        return sch.Schema(
            {
                col: type_mapper.from_string(typestr, nullable=nullable)
                for col, typestr, nullable in rows
            }
        )

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        name = util.gen_name(f"{self.name}_metadata")

        create_stmt = sge.Create(
            kind="VIEW",
            this=sg.table(name),
            expression=sg.parse_one(query, read=self.dialect),
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
        )
        drop_stmt = sge.Drop(kind="VIEW", this=sg.table(name), exists=True).sql(
            self.dialect
        )

        with self._safe_raw_sql(create_stmt):
            pass
        try:
            yield from self.get_schema(name).items()
        finally:
            with self._safe_raw_sql(drop_stmt):
                pass

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None and database != self.current_database:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support creating a schema in a different database"
            )
        sql = sge.Create(
            kind="SCHEMA", this=sg.table(name, catalog=database), exists=force
        )
        with self._safe_raw_sql(sql):
            pass

    def drop_schema(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        if database is not None and database != self.current_database:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support dropping a schema in a different database"
            )

        sql = sge.Drop(
            kind="SCHEMA",
            this=sg.table(name, catalog=database),
            exists=force,
            cascade=cascade,
        )
        with self._safe_raw_sql(sql):
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
    ):
        """Create a table in Postgres.

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
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if database is not None and database != self.current_database:
            raise com.UnsupportedOperationError(
                f"Creating tables in other databases is not supported by {self.name}"
            )
        else:
            database = None

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self._to_sqlglot(table)
        else:
            query = None

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
            for colname, typ in (schema or table.schema()).items()
        ]

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, catalog=database, quoted=self.compiler.quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table, expression=query).sql(self.dialect)
                cur.execute(insert_stmt)

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.dialect)
                )
                cur.execute(
                    f"ALTER TABLE IF EXISTS {table.sql(self.dialect)} RENAME TO {this.sql(self.dialect)}"
                )

        if schema is None:
            return self.table(name, schema=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def drop_table(
        self,
        name: str,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        if database is not None and database != self.current_database:
            raise com.UnsupportedOperationError(
                f"Droppping tables in other databases is not supported by {self.name}"
            )
        else:
            database = None
        drop_stmt = sg.exp.Drop(
            kind="TABLE",
            this=sg.table(
                name, db=schema, catalog=database, quoted=self.compiler.quoted
            ),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        cursor = con.cursor()

        try:
            # try to load hstore, uuid and ipaddress extensions
            with contextlib.suppress(psycopg2.ProgrammingError):
                extras.register_hstore(cursor)
            with contextlib.suppress(psycopg2.ProgrammingError):
                extras.register_uuid(conn_or_curs=cursor)
            with contextlib.suppress(psycopg2.ProgrammingError):
                extras.register_ipaddress(cursor)
        except Exception:
            cursor.close()
            raise

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            con.rollback()
            cursor.close()
            raise
        else:
            con.commit()
            return cursor

    def _to_sqlglot(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        table_expr = expr.as_table()
        conversions = {
            name: table_expr[name].as_ewkb()
            for name, typ in table_expr.schema().items()
            if typ.is_geospatial()
        }

        if conversions:
            table_expr = table_expr.mutate(**conversions)
        return super()._to_sqlglot(table_expr, limit=limit, params=params)
