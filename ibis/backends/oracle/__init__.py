"""The Oracle backend."""

from __future__ import annotations

import atexit
import contextlib
import sys
import warnings
from typing import TYPE_CHECKING, Any

import oracledb
import sqlglot as sg

from ibis import util

# Wow, this is truly horrible
# Get out your clippers, it's time to shave a yak.
#
# 1. snowflake-sqlalchemy doesn't support sqlalchemy 2.0
# 2. oracledb is only supported in sqlalchemy 2.0
# 3. Ergo, module hacking is required to avoid doing a silly amount of work
#    to create multiple lockfiles or port snowflake away from sqlalchemy
# 4. Also the version needs to be spoofed to be >= 7 or else the cx_Oracle
#    dialect barfs
oracledb.__version__ = oracledb.version = "7"

sys.modules["cx_Oracle"] = oracledb

import sqlalchemy as sa  # noqa: E402

import ibis.common.exceptions as exc  # noqa: E402
import ibis.expr.datatypes as dt  # noqa: E402
import ibis.expr.operations as ops  # noqa: E402
import ibis.expr.schema as sch  # noqa: E402
from ibis.backends.base.sql.alchemy import (  # noqa: E402
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.oracle.datatypes import OracleType  # noqa: E402
from ibis.backends.oracle.registry import operation_registry  # noqa: E402
from ibis.expr.rewrites import rewrite_sample  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable


class OracleExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _dialect_name = "oracle"
    _has_reduction_filter_syntax = False
    _require_order_by = (
        *AlchemyExprTranslator._require_order_by,
        ops.Reduction,
        ops.Lag,
        ops.Lead,
    )

    _forbids_frame_clause = (
        *AlchemyExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
    )

    _quote_column_names = True
    _quote_table_names = True

    type_mapper = OracleType


class OracleCompiler(AlchemyCompiler):
    translator_class = OracleExprTranslator
    support_values_syntax_in_select = False
    supports_indexed_grouping_keys = False
    null_limit = None
    rewrites = AlchemyCompiler.rewrites | rewrite_sample


class Backend(BaseAlchemyBackend):
    name = "oracle"
    compiler = OracleCompiler
    supports_create_or_replace = False
    supports_temporary_tables = True
    _temporary_prefix = "GLOBAL TEMPORARY"

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
        url = sa.engine.url.make_url(f"oracle://{user}:{password}@{dsn}")

        engine = sa.create_engine(
            url,
            poolclass=sa.pool.StaticPool,
            # We set the statement cache size to 0 because Oracle will otherwise
            # attempt to reuse prepared statements even if the type of the bound variable
            # has changed.
            # This is apparently accepted behavior.
            # https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_b.html#statement-caching-in-thin-and-thick-modes
            connect_args={"stmtcachesize": 0},
        )

        super().do_connect(engine)

        def normalize_name(name):
            if name is None:
                return None
            elif not name:
                return ""
            elif name.lower() == name:
                return sa.sql.quoted_name(name, quote=True)
            else:
                return name

        self.con.dialect.normalize_name = normalize_name

    def _from_url(self, url: str, **kwargs):
        return self.do_connect(user=url.username, password=url.password, dsn=url.host)

    @property
    def current_database(self) -> str:
        return self._scalar_query("SELECT * FROM global_name")

    def list_tables(self, like=None, database=None, schema=None):
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            (deprecated) The database to perform the list against.
        schema
            The schema to perform the list against.

            ::: {.callout-warning}
            ## `schema` refers to database hierarchy

            The `schema` parameter does **not** refer to the column names and
            types of `table`.
            :::
        """
        if database is not None:
            util.warn_deprecated(
                "database",
                instead="Use the `schema` keyword argument instead",
                as_of="7.1",
                removed_in="8.0",
            )
        schema = schema or database
        tables = self.inspector.get_table_names(schema=schema)
        views = self.inspector.get_view_names(schema=schema)
        return self._filter_with_like(tables + views, like)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        from sqlalchemy_views import CreateView, DropView

        name = util.gen_name("oracle_metadata")

        try:
            sg_expr = sg.parse_one(query, into=sg.exp.Table, dialect="oracle")
        except sg.ParseError:
            sg_expr = sg.parse_one(query, dialect="oracle")

        # If query is a table, adjust the query accordingly
        if isinstance(sg_expr, sg.exp.Table):
            sg_expr = sg.select("*").from_(sg_expr)

        query = sg_expr.sql(dialect="oracle")

        view = sa.table(name)
        create_view = CreateView(view, sa.text(query))
        drop_view = DropView(view, if_exists=False)

        t = sa.table(
            "all_tab_columns",
            sa.column("table_name"),
            sa.column("column_name"),
            sa.column("data_type"),
            sa.column("data_precision"),
            sa.column("data_scale"),
            sa.column("nullable"),
        )
        metadata_query = sa.select(
            t.c.column_name,
            t.c.data_type,
            t.c.data_precision,
            t.c.data_scale,
            (t.c.nullable == "Y").label("nullable"),
        ).where(t.c.table_name == name)

        with self.begin() as con:
            con.execute(create_view)
            try:
                results = con.execute(metadata_query).fetchall()
            finally:
                # drop the view no matter what
                con.execute(drop_view)

        for name, type_string, precision, scale, nullable in results:
            if precision is not None and scale is not None and precision != 0:
                typ = dt.Decimal(precision=precision, scale=scale, nullable=nullable)
            elif precision == 0:
                # TODO: how to disambiguate between int and float here without inspecting the value?
                typ = dt.float
            else:
                typ = OracleType.from_string(type_string, nullable=nullable)
            yield name, typ

    def _table_from_schema(
        self,
        name: str,
        schema: sch.Schema,
        temp: bool = False,
        database: str | None = None,
        **kwargs: Any,
    ) -> sa.Table:
        if temp:
            kwargs["oracle_on_commit"] = "PRESERVE ROWS"
        t = super()._table_from_schema(name, schema, temp, database, **kwargs)
        if temp:
            atexit.register(self._clean_up_tmp_table, t)
        return t

    def _clean_up_tmp_table(self, name: str) -> None:
        tmptable = self._get_sqla_table(name, autoload=False)
        with self.begin() as bind:
            # global temporary tables cannot be dropped without first truncating them
            #
            # https://stackoverflow.com/questions/32423397/force-oracle-drop-global-temp-table
            #
            # ignore DatabaseError exceptions because the table may not exist
            # because it's already been deleted
            with contextlib.suppress(sa.exc.DatabaseError):
                bind.exec_driver_sql(f'TRUNCATE TABLE "{tmptable.name}"')
            with contextlib.suppress(sa.exc.DatabaseError):
                tmptable.drop(bind=bind)

    def _clean_up_cached_table(self, op):
        self._clean_up_tmp_table(op.name)
