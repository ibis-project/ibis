"""RisingWave backend."""

from __future__ import annotations

import contextlib
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import psycopg2
import sqlglot as sg
import sqlglot.expressions as sge
from psycopg2 import extras

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    CanListCatalog,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoExampleLoader,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import TRUE, C, ColGen
from ibis.util import experimental

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


def data_and_encode_format(data_format, encode_format, encode_properties):
    res = ""
    if data_format is not None:
        res = res + " FORMAT " + data_format.upper()
    if encode_format is not None:
        res = res + " ENCODE " + encode_format.upper()
        if encode_properties is not None:
            res = res + " " + format_properties(encode_properties)
    return res


def format_properties(props):
    tokens = []
    for k, v in props.items():
        tokens.append(f"{k}='{v}'")
    return "( {} ) ".format(", ".join(tokens))


class Backend(
    SQLBackend,
    CanListCatalog,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoExampleLoader,
):
    name = "risingwave"
    compiler = sc.risingwave.compiler
    supports_python_udfs = False

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        database, *schema = url.path[1:].split("/", 1)
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        if schema:
            kwargs["schema"] = schema[0]
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

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

        from ibis.backends.risingwave.converter import RisingWavePandasData

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
        df = RisingWavePandasData.convert_table(df, schema)
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

    @util.experimental
    @classmethod
    def from_connection(cls, con: psycopg2.extensions.connection, /) -> Backend:
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
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
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

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        # http://dba.stackexchange.com/a/1304/58517
        cats = (
            sg.select(C.datname)
            .from_(sg.table("pg_database", db="pg_catalog"))
            .where(sg.not_(C.datistemplate))
        )
        with self._safe_raw_sql(cats) as cur:
            catalogs = list(map(itemgetter(0), cur))

        return self._filter_with_like(catalogs, like)

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
        self, name: str, /, *, catalog: str | None = None, force: bool = False
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
        /,
        *,
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
            this=sg.table(name),
            exists=force,
            cascade=cascade,
        )
        with self._safe_raw_sql(sql):
            pass

    def drop_table(
        self,
        name: str,
        /,
        *,
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

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create an Ibis client connected to RisingWave database.

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
            RisingWave schema to use. If `None`, use the default `search_path`.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_RISINGWAVE_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_RISINGWAVE_USER", "root")
        >>> password = os.environ.get("IBIS_TEST_RISINGWAVE_PASSWORD", "")
        >>> database = os.environ.get("IBIS_TEST_RISINGWAVE_DATABASE", "dev")
        >>> con = ibis.risingwave.connect(
        ...     database=database,
        ...     host=host,
        ...     user=user,
        ...     password=password,
        ...     port=4566,
        ... )
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

        self.con = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            options=(f"-csearch_path={schema}" * (schema is not None)) or None,
        )

        with self.begin() as cur:
            cur.execute("SET TIMEZONE = UTC")
            cur.execute("SET RW_IMPLICIT_FLUSH TO true;")

    def create_table(
        self,
        name: str,
        /,
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
        # TODO(Kexiang): add `append only`
        connector_properties: dict | None = None,
        data_format: str | None = None,
        encode_format: str | None = None,
        encode_properties: dict | None = None,
    ):
        """Create a table in RisingWave.

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
        connector_properties
            The properties of the sink connector, providing the connector settings to push to the downstream data sink.
            Refer https://docs.risingwave.com/docs/current/data-delivery/ for the required properties of different data sink.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.

        Returns
        -------
        Table
            Table expression
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        if connector_properties is not None and (
            encode_format is None or data_format is None
        ):
            raise com.UnsupportedOperationError(
                "When creating tables with connector, both encode_format and data_format are required"
            )

        properties = []

        if temp:
            raise com.UnsupportedOperationError(
                f"Creating temp tables is not supported by {self.name}"
            )

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
            this=table_expr,
            expressions=schema.to_sqlglot_column_defs(self.dialect),
        )

        if connector_properties is None:
            create_stmt = sge.Create(
                kind="TABLE",
                this=target,
                properties=sge.Properties(expressions=properties),
            )
        else:
            create_stmt = sge.Create(
                kind="TABLE",
                this=target,
                properties=sge.Properties.from_dict(connector_properties),
            )
            create_stmt = create_stmt.sql(self.dialect) + data_and_encode_format(
                data_format, encode_format, encode_properties
            )

        this = sg.table(name, db=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table_expr, expression=query).sql(
                    self.dialect
                )
                cur.execute(insert_stmt)

            if overwrite:
                self.drop_table(name, database=database, force=True)
                cur.execute(
                    f"ALTER TABLE {table_expr.sql(self.dialect)} RENAME TO {this.sql(self.dialect)}"
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(self.dialect),
            ),
        )
        create_stmt_sql = create_stmt.sql(self.dialect)

        df = op.data.to_frame()
        data = df.itertuples(index=False)
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)
            extras.execute_batch(cur, sql, data, 128)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        dbs = "SHOW SCHEMAS"

        with self._safe_raw_sql(dbs) as cur:
            databases = list(map(itemgetter(0), cur))

        return self._filter_with_like(databases, like)

    @experimental
    def create_materialized_view(
        self,
        name: str,
        /,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a materialized view. Materialized views can be accessed like a normal table.

        Parameters
        ----------
        name
            Materialized view name to Create.
        obj
            The select statement to materialize.
        database
            Name of the database where the view exists, if not the default
        overwrite
            Whether to overwrite the existing materialized view with the same name

        Returns
        -------
        Table
            Table expression
        """
        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, db=database, quoted=self.compiler.quoted)

        create_stmt = sge.Create(
            this=table,
            kind="MATERIALIZED VIEW",
            expression=self.compile(obj),
        )
        self._register_in_memory_tables(obj)

        with self._safe_raw_sql(create_stmt) as cur:
            if overwrite:
                target = sg.table(name, db=database).sql(self.dialect)

                self.drop_materialized_view(target, database=database, force=True)

                cur.execute(
                    f"ALTER MATERIALIZED VIEW {table.sql(self.dialect)} RENAME TO {target}"
                )

        return self.table(name, database=database)

    def drop_materialized_view(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a materialized view.

        Parameters
        ----------
        name
            Materialized view name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="MATERIALIZED VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def create_source(
        self,
        name: str,
        /,
        *,
        schema: ibis.Schema,
        database: str | None = None,
        connector_properties: dict,
        data_format: str,
        encode_format: str,
        encode_properties: dict | None = None,
        includes: dict[str, str] | None = None,
    ) -> ir.Table:
        """Creating a source.

        Parameters
        ----------
        name
            Source name to Create.
        schema
            The schema for the new Source.
        database
            Name of the database where the source exists, if not the default.
        connector_properties
            The properties of the source connector, providing the connector settings to access the upstream data source.
            Refer https://docs.risingwave.com/docs/current/data-ingestion/ for the required properties of different data source.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.
        includes
            A dict of `INCLUDE` clauses of the form `{field: alias, ...}`.
            Set value(s) to `None` if no alias is needed. Refer to https://docs.risingwave.com/docs/current/sql-create-source/ for more details.

        Returns
        -------
        Table
            Table expression
        """
        quoted = self.compiler.quoted
        table = sg.table(name, db=database, quoted=quoted)
        target = sge.Schema(
            this=table, expressions=schema.to_sqlglot_column_defs(self.dialect)
        )

        properties = sge.Properties.from_dict(connector_properties)
        properties.expressions.extend(
            sge.IncludeProperty(
                this=sg.to_identifier(include_type),
                alias=sg.to_identifier(column_name, quoted=quoted)
                if column_name
                else None,
            )
            for include_type, column_name in (includes or {}).items()
        )

        create_stmt = sge.Create(kind="SOURCE", this=target, properties=properties)

        create_stmt = create_stmt.sql(self.dialect) + data_and_encode_format(
            data_format, encode_format, encode_properties
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, database=database)

    def drop_source(
        self, name: str, /, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a Source.

        Parameters
        ----------
        name
            Source name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the source does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="SOURCE",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def create_sink(
        self,
        name: str,
        /,
        *,
        sink_from: str | None = None,
        connector_properties: dict | None = None,
        obj: ir.Table | None = None,
        database: str | None = None,
        data_format: str | None = None,
        encode_format: str | None = None,
        encode_properties: dict | None = None,
    ) -> None:
        """Creating a sink.

        Parameters
        ----------
        name
            Sink name to Create.
        sink_from
            The table or materialized view name to sink from. Only one of `sink_from` or `obj` can be
            provided.
        connector_properties
            The properties of the sink connector, providing the connector settings to push to the downstream data sink.
            Refer https://docs.risingwave.com/docs/current/data-delivery/ for the required properties of different data sink.
        obj
            An Ibis table expression that will be used to extract the schema and the data of the new table. Only one of `sink_from` or `obj` can be provided.
        database
            Name of the database where the source exists, if not the default.
        data_format
            The data format for the new source, e.g., "PLAIN". data_format and encode_format must be specified at the same time.
        encode_format
            The encode format for the new source, e.g., "JSON". data_format and encode_format must be specified at the same time.
        encode_properties
            The properties of encode format, providing information like schema registry url. Refer https://docs.risingwave.com/docs/current/sql-create-source/ for more details.
        """
        table = sg.table(name, db=database, quoted=self.compiler.quoted)
        if sink_from is None and obj is None:
            raise ValueError("Either `sink_from` or `obj` must be specified")
        if sink_from is not None and obj is not None:
            raise ValueError("Only one of `sink_from` or `obj` can be specified")

        if (encode_format is None) != (data_format is None):
            raise com.UnsupportedArgumentError(
                "When creating sinks, both encode_format and data_format must be provided, or neither should be"
            )

        if sink_from is not None:
            create_stmt = f"CREATE SINK {table.sql(self.dialect)} FROM {sink_from}"
        else:
            create_stmt = sge.Create(
                this=table,
                kind="SINK",
                expression=self.compile(obj),
            ).sql(self.dialect)
        create_stmt = (
            create_stmt
            + " WITH "
            + format_properties(connector_properties)
            + data_and_encode_format(data_format, encode_format, encode_properties)
        )
        with self._safe_raw_sql(create_stmt):
            pass

    def drop_sink(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a Sink.

        Parameters
        ----------
        name
            Sink name to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the source does not exist.
        """
        src = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="SINK",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    @property
    def _session_temp_db(self) -> str | None:
        # Return `None`, because RisingWave does not implement temp tables like
        # Postgres
        return None

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        cursor = con.cursor()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            con.rollback()
            cursor.close()
            raise
        else:
            con.commit()
            return cursor
