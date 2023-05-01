"""The Oracle backend."""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import sqlalchemy as sa

import ibis.backends.oracle.datatypes as odt
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.oracle.registry import operation_registry


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


class OracleCompiler(AlchemyCompiler):
    translator_class = OracleExprTranslator
    support_values_syntax_in_select = False
    supports_indexed_grouping_keys = False


class Backend(BaseAlchemyBackend):
    name = 'oracle'
    compiler = OracleCompiler
    supports_create_or_replace = False

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
            f"oracle+oracledb://{user}:{password}@{host}:{port}/{database}"
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
            connect_args={
                "service_name": database,
            },
            isolation_level="READ COMMITTED",
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                try:
                    cur.execute("COMMIT")
                except sa.exc.OperationalError:
                    warnings.warn("RUH ROH NO COMMIT")

        res = super().do_connect(engine)

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
        return res

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
                typ = odt.parse(type_code).copy(nullable=is_nullable)
            yield name, typ

    def _table_from_schema(
        self,
        name: str,
        schema: sch.Schema,
        database: str | None = None,
        temp: bool = False,
    ) -> sa.Table:
        table = super()._table_from_schema(
            name, schema=schema, database=database, temp=temp
        )
        if temp:
            # Oracle complains about this missing `GLOBAL` keyword so we add it
            # in here.  Not sure if this is always necessary or only some of the
            # time
            table._prefixes.insert(0, "GLOBAL")
        return table

    # TODO: figure out when/how oracle drops temp tables
    # def list_tables(self, like=None, database=None):
    #     tables = self.inspector.get_table_names(schema=database)
    #     views = self.inspector.get_view_names(schema=database)
    #     return self._filter_with_like(tables + views, like)
