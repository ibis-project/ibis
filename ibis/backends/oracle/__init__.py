"""The Oracle backend."""

from __future__ import annotations

import atexit
import contextlib
import sys

import oracledb

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

from typing import TYPE_CHECKING, Any, Iterable  # noqa: E402

import sqlalchemy as sa  # noqa: E402

import ibis.expr.datatypes as dt  # noqa: E402
import ibis.expr.operations as ops  # noqa: E402
from ibis.backends.base.sql.alchemy import (  # noqa: E402
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.oracle.datatypes import (  # noqa: E402
    OracleType,
    parse,
)
from ibis.backends.oracle.registry import operation_registry  # noqa: E402

if TYPE_CHECKING:
    import ibis.expr.schema as sch


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
        database: str | None = "FREE",
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
            Database to connect to
        """
        url = sa.engine.url.make_url(
            f"oracle://{user}:{password}@{host}:{port}/{database}"
        )

        # Creating test DB and user
        # The ORACLE_DB env-var needs to be set in the docker-compose.yml file
        # Then, after the container is running, exec in and run (from `/opt/oracle`)
        # ./createAppUser user pass ORACLE_DB
        # where ORACLE_DB is the same name you used in the docker-compose file.

        # ORACLE IS VERY CONFUSING
        # SID -- instance identifier -- meant to distinguish oracle instances running on the same machine
        # TABLESPACE -- logical grouping of tables and views, unclear how different from DATABASE
        # DATABASE can be assigned (defaults?) to a tablespace
        #
        # sqlplus ibis/ibis@localhost:1521/IBIS_TESTING
        # for connecting from docker exec
        #
        # for current session parameters
        # select * from nls_session_parameters;
        #
        # alter session parameter e.g.
        # alter session set nls_timestamp_format='YYYY-MM-DD HH24:MI:SS.FF3'
        #
        # see user tables
        # select table_name from user_tables

        self.database_name = database  # not sure what should go here

        # Note: for the moment, we need to pass the `database` in to the `make_url` call
        # AND specify it here as the `service_name`.  I don't know why.
        engine = sa.create_engine(
            url,
            poolclass=sa.pool.StaticPool,
            # We set the statement cache size to 0 because Oracle will otherwise
            # attempt to reuse prepared statements even if the type of the bound variable
            # has changed.
            # This is apparently accepted behavior.
            # https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_b.html#statement-caching-in-thin-and-thick-modes
            connect_args={"service_name": database, "stmtcachesize": 0},
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

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        query = f"SELECT * FROM ({query.strip(';')}) FETCH FIRST 0 ROWS ONLY"
        with self.begin() as con, con.connection.cursor() as cur:
            result = cur.execute(query)
            desc = result.description

        for name, type_code, _, _, precision, scale, is_nullable in desc:
            if precision is not None and scale is not None and precision != 0:
                typ = dt.Decimal(precision=precision, scale=scale, nullable=is_nullable)
            elif precision == 0:
                # TODO: how to disambiguate between int and float here without inspecting the value?
                typ = dt.float
            else:
                typ = parse(type_code).copy(nullable=is_nullable)
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
