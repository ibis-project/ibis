"""The Oracle backend."""

from __future__ import annotations

import contextlib
import re
import warnings
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import unquote_plus

import oracledb
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanListDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    PyArrowExampleLoader,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR, C

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


def metadata_row_to_type(
    *, type_mapper, type_string, precision, scale, nullable
) -> dt.DataType:
    """Convert a row from an Oracle metadata table to an Ibis type."""
    # See
    # https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Data-Types.html#GUID-0BC16006-32F1-42B1-B45E-F27A494963FF
    # for details
    #
    # NUMBER(null, null) --> NUMBER(38) -> NUMBER(38, 0)
    # (null, null) --> from_string()
    if type_string == "NUMBER" and precision is None and not scale:
        typ = dt.Int64(nullable=nullable)

    # (null, 0) --> INT
    # (null, 3), (null, 6), (null, 9) --> from_string() - TIMESTAMP(3)/(6)/(9)
    elif precision is None and (scale is not None and scale == 0):
        typ = dt.Int64(nullable=nullable)

    # NUMBER(*, 0) --> INT
    # (*, 0) --> from_string() - INTERVAL DAY(3) TO SECOND(0)
    elif (
        type_string == "NUMBER"
        and precision is not None
        and (scale is not None and scale == 0)
    ):
        typ = dt.Int64(nullable=nullable)

    # NUMBER(*, > 0) --> DECIMAL
    # (*, > 0) --> from_string() - INTERVAL DAY(3) TO SECOND(2)
    elif (
        type_string == "NUMBER"
        and precision is not None
        and (scale is not None and scale > 0)
    ):
        typ = dt.Decimal(precision=int(precision), scale=int(scale), nullable=nullable)

    else:
        typ = type_mapper.from_string(type_string, nullable=nullable)
    return typ


class Backend(
    SQLBackend,
    CanListDatabase,
    HasCurrentDatabase,
    HasCurrentCatalog,
    PyArrowExampleLoader,
):
    name = "oracle"
    compiler = sc.oracle.compiler
    supports_temporary_tables = True

    @cached_property
    def version(self):
        matched = re.search(r"(\d+)\.(\d+)\.(\d+)", self.con.version)
        return ".".join(matched.groups())

    def do_connect(
        self,
        *,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 1521,
        database: str | None = None,
        sid: str | None = None,
        service_name: str | None = None,
        dsn: str | None = None,
        **_: Any,
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        user
            Username
        password
            Password
        host
            Hostname
        port
            Port
        database
            Used as an Oracle service name if provided.
        sid
            Unique name of an Oracle Instance, used to construct a DSN if
            provided.
        service_name
            Oracle service name, used to construct a DSN if provided.  Only one
            of database and service_name should be provided.
        dsn
            An Oracle Data Source Name.  If provided, overrides all other
            connection arguments except username and password.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_ORACLE_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_ORACLE_USER", "ibis")
        >>> password = os.environ.get("IBIS_TEST_ORACLE_PASSWORD", "ibis")
        >>> database = os.environ.get("IBIS_TEST_ORACLE_DATABASE", "IBIS_TESTING")
        >>> con = ibis.oracle.connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int64
          bool_col        int64
          tinyint_col     int64
          smallint_col    int64
          int_col         int64
          bigint_col      int64
          float_col       float64
          double_col      float64
          date_string_col string
          string_col      string
          timestamp_col   timestamp(3)
          year            int64
          month           int64
        """
        # SID: unique name of an INSTANCE running an oracle process (a single, identifiable machine)
        # service name: an ALIAS to one (or many) individual instances that can
        # be hotswapped without the client knowing / caring
        if dsn is not None and (
            database is not None or sid is not None or service_name is not None
        ):
            warnings.warn(
                "Oracle DSN provided, overriding additional provided connection arguments"
            )

        if service_name is not None and database is not None:
            raise exc.IbisInputError(
                "Values provided for both service_name and database. "
                "Both of these values map to an Oracle service_name, "
                "please provide only one of them."
            )

        if service_name is None and database is not None:
            service_name = database

        if dsn is None:
            dsn = oracledb.makedsn(host, port, service_name=service_name, sid=sid)

        # We set the statement cache size to 0 because Oracle will otherwise
        # attempt to reuse prepared statements even if the type of the bound variable
        # has changed.
        # This is apparently accepted behavior.
        # https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_b.html#statement-caching-in-thin-and-thick-modes
        self.con = oracledb.connect(dsn, user=user, password=password, stmtcachesize=0)

        self._post_connect()

    @util.experimental
    @classmethod
    def from_connection(cls, con: oracledb.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to an Oracle database.

        Parameters
        ----------
        con
            An existing connection to an Oracle database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def _post_connect(self) -> None:
        # turn on autocommit
        # TODO: it would be great if this worked but it doesn't seem to do the trick
        # I had to hack in the commit lines to the compiler
        # self.con.autocommit = True

        # Set to ensure decimals come back as decimals
        oracledb.defaults.fetch_decimals = True

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database := url.path.removeprefix("/"):
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(STAR).from_("global_name")) as cur:
            [(catalog,)] = cur.fetchall()
        return catalog

    @property
    def current_database(self) -> str:
        # databases correspond to users, other than that there's
        # no notion of a database inside a catalog for oracle
        with self._safe_raw_sql(sg.select("user").from_("dual")) as cur:
            [(database,)] = cur.fetchall()
        return database

    @contextlib.contextmanager
    def begin(self):
        con = self.con
        cur = con.cursor()
        try:
            yield cur
        except Exception:
            con.rollback()
            raise
        else:
            con.commit()
        finally:
            cur.close()

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

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

    def list_tables(
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        if database is not None:
            table_loc = database
        else:
            table_loc = self.con.username.upper()

        table_loc = self._to_sqlglot_table(table_loc)

        dialect = self.dialect

        # Deeply frustrating here where if we call `convert` on `table_loc`,
        # which is a sg.exp.Table, the quotes are rendered as double-quotes
        # which are invalid. So, we unquote the database name here.
        def unquote(node):
            if isinstance(node, sg.exp.Identifier):
                return sg.to_identifier(node.name, quoted=False)
            return node

        conditions = C.owner.eq(sge.convert(table_loc.transform(unquote).sql(dialect)))

        tables = (
            sg.select(C.table_name)
            .from_("all_tables")
            .distinct()
            .where(conditions)
            .union(
                sg.select(C.view_name).from_("all_views").distinct().where(conditions)
            )
        )

        with self._safe_raw_sql(tables) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        if catalog is not None:
            raise exc.UnsupportedArgumentError(
                "No cross-catalog schema access in Oracle"
            )

        query = sg.select("username").from_("all_users").order_by("username")

        with self._safe_raw_sql(query) as con:
            schemata = list(map(itemgetter(0), con))

        return self._filter_with_like(schemata, like)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        if database is None:
            database = self.con.username.upper()
        stmt = (
            sg.select(
                C.column_name,
                C.data_type,
                C.data_precision,
                C.data_scale,
                C.nullable.eq(sge.convert("Y")).as_("nullable"),
            )
            .from_(sg.table("all_tab_columns"))
            .where(
                C.table_name.eq(sge.convert(name)),
                C.owner.eq(sge.convert(database)),
            )
            .order_by(C.column_id)
        )
        with self._safe_raw_sql(stmt) as cur:
            results = cur.fetchall()

        if not results:
            raise exc.TableNotFound(name)

        type_mapper = self.compiler.type_mapper
        fields = {
            name: metadata_row_to_type(
                type_mapper=type_mapper,
                type_string=type_string,
                precision=precision,
                scale=scale,
                nullable=nullable,
            )
            for name, type_string, precision, scale, nullable in results
        }

        return sch.Schema(fields)

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
    ):
        """Create a table in Oracle.

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

        initial_table = sg.table(temp_name, db=database, quoted=self.compiler.quoted)
        target = sge.Schema(
            this=initial_table,
            expressions=(schema or table.schema()).to_sqlglot_column_defs(self.dialect),
        )

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        # This is the same table as initial_table unless overwrite == True
        final_table = sg.table(name, db=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=initial_table, expression=query).sql(
                    self.name
                )
                cur.execute(insert_stmt)

            if overwrite:
                self.drop_table(final_table.name, database=final_table.db, force=True)
                cur.execute(
                    f"ALTER TABLE IF EXISTS {initial_table.sql(self.name)} RENAME TO {final_table.sql(self.name)}"
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
        /,
        *,
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        table_loc = self._to_sqlglot_table(database or None)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        table = sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted)

        with self.begin() as bind:
            # global temporary tables cannot be dropped without first truncating them
            #
            # https://stackoverflow.com/questions/32423397/force-oracle-drop-global-temp-table
            #
            # ignore DatabaseError exceptions because the table may not exist
            # because it's already been deleted
            with contextlib.suppress(oracledb.DatabaseError):
                bind.execute(f"TRUNCATE TABLE {table.sql(self.name)}")

        super().drop_table(name, database=(catalog, db), force=force)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(self.dialect),
            ),
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
        ).sql(self.name)

        data = op.data.to_frame().replace(float("nan"), None)
        insert_stmt = self._build_insert_template(
            name, schema=schema, placeholder=":{i:d}"
        )
        with self.begin() as cur:
            cur.execute(create_stmt)
            for start, end in util.chunks(len(data), chunk_size=128):
                cur.executemany(
                    insert_stmt, list(data.iloc[start:end].itertuples(index=False))
                )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        name = util.gen_name("oracle_metadata")
        dialect = self.name

        try:
            sg_expr = sg.parse_one(query, into=sg.exp.Table, dialect=dialect)
        except sg.ParseError:
            sg_expr = sg.parse_one(query, dialect=dialect)

        # If query is a table, adjust the query accordingly
        if isinstance(sg_expr, sg.exp.Table):
            sg_expr = sg.select(STAR).from_(sg_expr)

        # TODO(gforsyth): followup -- this should probably be made a default
        # transform for quoting backends
        def transformer(node):
            if isinstance(node, sg.exp.Table):
                return sg.table(node.name, quoted=True)
            elif isinstance(node, sg.exp.Column):
                return sg.column(col=node.name, quoted=True)
            return node

        sg_expr = sg_expr.transform(transformer)

        this = sg.table(name, quoted=True)
        create_view = sg.exp.Create(kind="VIEW", this=this, expression=sg_expr).sql(
            dialect
        )
        drop_view = sg.exp.Drop(kind="VIEW", this=this).sql(dialect)

        metadata_query = (
            sg.select(
                C.column_name,
                C.data_type,
                C.data_precision,
                C.data_scale,
                C.nullable.eq(sge.convert("Y")),
            )
            .from_("all_tab_columns")
            .where(C.table_name.eq(sge.convert(name)))
            .order_by(C.column_id)
            .sql(dialect)
        )

        with self.begin() as con:
            con.execute(create_view)
            try:
                results = con.execute(metadata_query).fetchall()
            finally:
                # drop the view no matter what
                con.execute(drop_view)

        schema = {}

        # TODO: hand all this off to the type mapper
        type_mapper = self.compiler.type_mapper
        for name, type_string, precision, scale, nullable in results:
            typ = metadata_row_to_type(
                type_mapper=type_mapper,
                type_string=type_string,
                precision=precision,
                scale=scale,
                nullable=nullable,
            )
            schema[name] = typ

        return sch.Schema(schema)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        # TODO(gforsyth): this can probably be generalized a bit and put into
        # the base backend (or a mixin)
        import pandas as pd

        from ibis.backends.oracle.converter import OraclePandasData

        df = pd.DataFrame.from_records(cursor, columns=schema.names, coerce_float=True)
        return OraclePandasData.convert_table(df, schema)

    def _clean_up_tmp_table(self, name: str) -> None:
        dialect = self.dialect

        ident = sg.to_identifier(name, quoted=self.compiler.quoted)

        truncate = sge.TruncateTable(expressions=[ident]).sql(dialect)
        drop = sge.Drop(kind="TABLE", this=ident).sql(dialect)

        with self.begin() as bind:
            # global temporary tables cannot be dropped without first truncating them
            #
            # https://stackoverflow.com/questions/32423397/force-oracle-drop-global-temp-table
            #
            # ignore DatabaseError exceptions because the table may not exist
            # because it's already been deleted
            with contextlib.suppress(oracledb.DatabaseError):
                bind.execute(truncate)
            with contextlib.suppress(oracledb.DatabaseError):
                bind.execute(drop)

    _drop_cached_table = _clean_up_tmp_table

    def _make_memtable_finalizer(self, name: str) -> Callable[..., None]:
        dialect = self.dialect

        ident = sg.to_identifier(name, quoted=self.compiler.quoted)

        truncate = sge.TruncateTable(expressions=[ident]).sql(dialect)
        drop = sge.Drop(kind="TABLE", this=ident).sql(dialect)

        def finalizer(con=self.con, name: str = name) -> None:
            cursor = con.cursor()
            try:
                # global temporary tables cannot be dropped without first truncating them
                #
                # https://stackoverflow.com/questions/32423397/force-oracle-drop-global-temp-table
                #
                # ignore DatabaseError exceptions because the table may not exist
                # because it's already been deleted
                with contextlib.suppress(oracledb.DatabaseError):
                    cursor.execute(truncate)
                with contextlib.suppress(oracledb.DatabaseError):
                    cursor.execute(drop)
            except Exception:
                con.rollback()
                raise
            else:
                con.commit()
            finally:
                cursor.close()

        return finalizer
