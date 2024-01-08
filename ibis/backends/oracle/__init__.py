"""The Oracle backend."""

from __future__ import annotations

import atexit
import contextlib
import re
import warnings
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import oracledb
import sqlglot as sg
import sqlglot.expressions as sge

# Wow, this is truly horrible
# Get out your clippers, it's time to shave a yak.
#
# 1. oracledb is only supported in sqlalchemy 2.0
# 2. Ergo, module hacking is required to avoid doing a silly amount of work
#    to create multiple lockfiles or port snowflake away from sqlalchemy
# 3. Also the version needs to be spoofed to be >= 7 or else the cx_Oracle
#    dialect barfs
import ibis
import ibis.common.exceptions as exc  # noqa: E402
import ibis.expr.datatypes as dt  # noqa: E402
import ibis.expr.operations as ops  # noqa: E402
import ibis.expr.schema as sch  # noqa: E402
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sqlglot import STAR, SQLGlotBackend  # noqa: E402
from ibis.backends.base.sqlglot.compiler import TRUE, C
from ibis.backends.oracle.compiler import OracleCompiler

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import pyrrow as pa


# class OracleExprTranslator(AlchemyExprTranslator):
#     _registry = operation_registry.copy()
#     _rewrites = AlchemyExprTranslator._rewrites.copy()
#     _dialect_name = "oracle"
#     _has_reduction_filter_syntax = False
#     _require_order_by = (
#         *AlchemyExprTranslator._require_order_by,
#         ops.Reduction,
#         ops.Lag,
#         ops.Lead,
#     )

#     _forbids_frame_clause = (
#         *AlchemyExprTranslator._forbids_frame_clause,
#         ops.Lag,
#         ops.Lead,
#     )

#     _quote_column_names = True
#     _quote_table_names = True

#     type_mapper = OracleType


# class OracleCompiler(AlchemyCompiler):
#     translator_class = OracleExprTranslator
#     support_values_syntax_in_select = False
#     supports_indexed_grouping_keys = False
#     null_limit = None
#     rewrites = AlchemyCompiler.rewrites | rewrite_sample


class Backend(SQLGlotBackend):
    name = "oracle"
    compiler = OracleCompiler()

    # supports_create_or_replace = False
    # supports_temporary_tables = True
    # _temporary_prefix = "GLOBAL TEMPORARY"
    #
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

    def _from_url(self, url: str, **kwargs):
        return self.do_connect(user=url.username, password=url.password, dsn=url.host)

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(STAR).from_("global_name")) as cur:
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
        self, like: str | None = None, schema: str | None = None
    ) -> list[str]:
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        schema
            The schema to perform the list against.
        """
        conditions = [TRUE]

        if schema is not None:
            conditions = C.table_schema.eq(sge.convert(schema))

        # TODO: add support for other tables
        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("user_tables"))
            .distinct()
            .where(*conditions)
            .sql(self.name, pretty=True)
        )

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def get_schema(
        self, name: str, schema: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(name, db=schema, catalog=database, quoted=True)

        stmt = (
            sg.select(
                "column_name",
                "data_type",
                sg.column("nullable").eq(sge.convert("Y")).as_("nullable"),
            )
            .from_(sg.table("user_tab_columns"))
            .where(sg.column("table_name").eq(sge.convert(name)))
        )
        with self._safe_raw_sql(stmt) as cur:
            result = cur.fetchall()

        breakpoint()
        type_mapper = self.compiler.type_mapper
        fields = {
            name: type_mapper.from_string(type_string, nullable=nullable)
            for name, type_string, nullable, *_ in result
        }

        return sch.Schema(fields)

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
                insert_stmt = sge.Insert(this=table, expression=query).sql(self.name)
                cur.execute(insert_stmt).fetchall()

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.name)
                )
                cur.execute(
                    f"ALTER TABLE IF EXISTS {table.sql(self.name)} RENAME TO {this.sql(self.name)}"
                )

        if schema is None:
            return self.table(name, schema=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        name = util.gen_name("oracle_metadata")
        dialect = self.name

        try:
            sg_expr = sg.parse_one(query, into=sg.exp.Table, dialect=dialect)
        except sg.ParseError:
            sg_expr = sg.parse_one(query, dialect=dialect)

        # If query is a table, adjust the query accordingly
        if isinstance(sg_expr, sg.exp.Table):
            sg_expr = sg.select(STAR).from_(sg_expr)

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
                C.nullable.eq("Y"),
            )
            .from_("all_tab_columns")
            .where(C.table_name.eq(name))
            .order_by(C.column_id)
            .sql(dialect)
        )

        with self.begin() as con:
            con.exec_driver_sql(create_view)
            try:
                results = con.exec_driver_sql(metadata_query).fetchall()
            finally:
                # drop the view no matter what
                con.exec_driver_sql(drop_view)

        for name, type_string, precision, scale, nullable in results:
            # NUMBER(null, null) --> FLOAT
            # (null, null) --> from_string()
            if type_string == "NUMBER" and precision is None and scale is None:
                typ = dt.Float64(nullable=nullable)

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
                typ = dt.Decimal(precision=precision, scale=scale, nullable=nullable)

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
