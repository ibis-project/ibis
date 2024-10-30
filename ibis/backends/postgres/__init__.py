"""PostgreSQL backend."""

from __future__ import annotations

import contextlib
import inspect
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge
from pandas.api.types import is_float_dtype

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, CanListCatalog
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import TRUE, C, ColGen

if TYPE_CHECKING:
    from collections.abc import Callable
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import psycopg2
    import pyarrow as pa


class Backend(SQLBackend, CanListCatalog, CanCreateDatabase):
    name = "postgres"
    compiler = sc.postgres.compiler
    supports_python_udfs = True

    def _from_url(self, url: ParseResult, **kwargs):
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
        database, *schema = url.path[1:].split("/", 1)
        connect_args = {
            "user": url.username,
            "password": unquote_plus(url.password or ""),
            "host": url.hostname,
            "database": database or "",
            "schema": schema[0] if schema else "",
            "port": url.port,
        }

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

        if "port" in kwargs and kwargs["port"] is None:
            del kwargs["port"]

        return self.connect(**kwargs)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        from psycopg2.extras import execute_batch

        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted
        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot(self.dialect),
            ),
            properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
        )
        create_stmt_sql = create_stmt.sql(self.dialect)

        df = op.data.to_frame()
        # nan gets compiled into 'NaN'::float which throws errors in non-float columns
        # In order to hold NaN values, pandas automatically converts integer columns
        # to float columns if there are NaN values in them. Therefore, we need to convert
        # them to their original dtypes (that support pd.NA) to figure out which columns
        # are actually non-float, then fill the NaN values in those columns with None.
        convert_df = df.convert_dtypes()
        for col in convert_df.columns:
            if not is_float_dtype(convert_df[col]):
                df[col] = df[col].replace(float("nan"), None)

        data = df.itertuples(index=False)
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )

        with self.begin() as cur:
            cur.execute(create_stmt_sql)
            execute_batch(cur, sql, data, 128)

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
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_POSTGRES_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_POSTGRES_USER", "postgres")
        >>> password = os.environ.get("IBIS_TEST_POSTGRES_PASSWORD", "postgres")
        >>> database = os.environ.get("IBIS_TEST_POSTGRES_DATABASE", "ibis_testing")
        >>> con = ibis.postgres.connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int32
          bool_col        boolean
          tinyint_col     int16
          smallint_col    int16
          int_col         int32
          bigint_col      int64
          float_col       float32
          double_col      float64
          date_string_col string
          string_col      string
          timestamp_col   timestamp(6)
          year            int32
          month           int32
        """
        import psycopg2
        import psycopg2.extras

        psycopg2.extras.register_default_json(loads=lambda x: x)

        self.con = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            options=(f"-csearch_path={schema}" * (schema is not None)) or None,
            **kwargs,
        )

        self._post_connect()

    @util.experimental
    @classmethod
    def from_connection(cls, con: psycopg2.extensions.connection) -> Backend:
        """Create an Ibis client from an existing connection to a PostgreSQL database.

        Parameters
        ----------
        con
            An existing connection to a PostgreSQL database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def _post_connect(self) -> None:
        with self.begin() as cur:
            cur.execute("SET TIMEZONE = UTC")

    @property
    def _session_temp_db(self) -> str | None:
        # Postgres doesn't assign the temporary table database until the first
        # temp table is created in a given session.
        # Before that temp table is created, this will return `None`
        # After a temp table is created, it will return `pg_temp_N` where N is
        # some integer
        res = self.raw_sql(
            "select nspname from pg_namespace where oid = pg_my_temp_schema()"
        ).fetchone()
        if res is not None:
            return res[0]
        return res

    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        """List the tables in the database.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of tables is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database.
        """

        if database is not None:
            table_loc = database
        else:
            table_loc = (self.current_catalog, self.current_database)

        table_loc = self._to_sqlglot_table(table_loc)

        conditions = [TRUE]

        if (db := table_loc.args["db"]) is not None:
            db.args["quoted"] = False
            db = db.sql(dialect=self.name)
            conditions.append(C.table_schema.eq(sge.convert(db)))
        if (catalog := table_loc.args["catalog"]) is not None:
            catalog.args["quoted"] = False
            catalog = catalog.sql(dialect=self.name)
            conditions.append(C.table_catalog.eq(sge.convert(catalog)))

        sql = (
            sg.select("table_name")
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql(self.dialect)
        )

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        # Include temporary tables only if no database has been explicitly specified
        # to avoid temp tables showing up in all calls to `list_tables`
        if db == "public":
            out += self._fetch_temp_tables()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def _fetch_temp_tables(self):
        # postgres temporary tables are stored in a separate schema
        # so we need to independently grab them and return them along with
        # the existing results

        sql = (
            sg.select("table_name")
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(C.table_type.eq(sge.convert("LOCAL TEMPORARY")))
            .sql(self.dialect)
        )

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return out

    def list_catalogs(self, like=None) -> list[str]:
        # http://dba.stackexchange.com/a/1304/58517
        cats = (
            sg.select(C.datname)
            .from_(sg.table("pg_database", db="pg_catalog"))
            .where(sg.not_(C.datistemplate))
        )
        with self._safe_raw_sql(cats) as cur:
            catalogs = list(map(itemgetter(0), cur))

        return self._filter_with_like(catalogs, like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        dbs = sg.select(C.schema_name).from_(
            sg.table("schemata", db="information_schema")
        )
        with self._safe_raw_sql(dbs) as cur:
            databases = list(map(itemgetter(0), cur))

        return self._filter_with_like(databases, like)

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(sg.func("current_database"))) as cur:
            (db,) = cur.fetchone()
        return db

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(sg.func("current_schema"))) as cur:
            (schema,) = cur.fetchone()
        return schema

    def function(self, name: str, *, database: str | None = None) -> Callable:
        n = ColGen(table="n")
        p = ColGen(table="p")
        f = self.compiler.f

        predicates = [p.proname.eq(sge.convert(name))]

        if database is not None:
            predicates.append(n.nspname.rlike(sge.convert(f"^({database})$")))

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
            .where(*predicates)
        )

        def split_name_type(arg: str) -> tuple[str, dt.DataType]:
            name, typ = arg.split(" ", 1)
            return name, self.compiler.type_mapper.from_string(typ)

        with self._safe_raw_sql(query) as cur:
            rows = cur.fetchall()

        if not rows:
            name = f"{database}.{name}" if database else name
            raise exc.MissingUDFError(name)
        elif len(rows) > 1:
            raise exc.AmbiguousUDFError(name)

        [(raw_return_type, signature)] = rows
        return_type = self.compiler.type_mapper.from_string(raw_return_type)
        signature = list(map(split_name_type, signature))

        # dummy callable
        def fake_func(*args, **kwargs): ...

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
        op = ops.udf.scalar.builtin(fake_func, database=database)
        return op

    def get_schema(
        self,
        name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ):
        a = ColGen(table="a")
        c = ColGen(table="c")
        n = ColGen(table="n")

        format_type = self.compiler.f["pg_catalog.format_type"]

        # If no database is specified, assume the current database
        db = database or self.current_database

        dbs = [sge.convert(db)]

        # If a database isn't specified, then include temp tables in the
        # returned values
        if database is None and (temp_table_db := self._session_temp_db) is not None:
            dbs.append(sge.convert(temp_table_db))

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
                n.nspname.isin(*dbs),
                c.relname.eq(sge.convert(name)),
            )
            .order_by(a.attnum)
        )

        type_mapper = self.compiler.type_mapper

        with self._safe_raw_sql(type_info) as cur:
            rows = cur.fetchall()

        if not rows:
            raise com.TableNotFound(name)

        return sch.Schema(
            {
                col: type_mapper.from_string(typestr, nullable=nullable)
                for col, typestr, nullable in rows
            }
        )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
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
            return self.get_schema(name)
        finally:
            with self._safe_raw_sql(drop_stmt):
                pass

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None and catalog != self.current_catalog:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support creating a database in a different catalog"
            )
        sql = sge.Create(
            kind="SCHEMA", this=sg.table(name, catalog=catalog), exists=force
        )
        with self._safe_raw_sql(sql):
            pass

    def drop_database(
        self,
        name: str,
        catalog: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        if catalog is not None and catalog != self.current_catalog:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support dropping a database in a different catalog"
            )

        sql = sge.Drop(
            kind="SCHEMA",
            this=sg.table(name, catalog=catalog),
            exists=force,
            cascade=cascade,
        )
        with self._safe_raw_sql(sql):
            pass

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

        table_expr = sg.table(temp_name, db=database, quoted=self.compiler.quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot(self.dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table_expr, expression=query).sql(
                    self.dialect
                )
                cur.execute(insert_stmt)

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.dialect)
                )
                cur.execute(
                    f"ALTER TABLE IF EXISTS {table_expr.sql(self.dialect)} RENAME TO {this.sql(self.dialect)}"
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
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        import psycopg2
        import psycopg2.extras

        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        cursor = con.cursor()

        try:
            # try to load hstore, uuid and ipaddress extensions
            with contextlib.suppress(psycopg2.ProgrammingError):
                psycopg2.extras.register_hstore(cursor)
            with contextlib.suppress(psycopg2.ProgrammingError):
                psycopg2.extras.register_uuid(conn_or_curs=cursor)
            with contextlib.suppress(psycopg2.ProgrammingError):
                psycopg2.extras.register_ipaddress(cursor)
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
