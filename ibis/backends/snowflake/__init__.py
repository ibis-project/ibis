from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterable

import snowflake.connector as sfc
import sqlalchemy as sa
import toolz
from snowflake.sqlalchemy import URL

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.snowflake.datatypes import parse
from ibis.backends.snowflake.registry import operation_registry

if TYPE_CHECKING:
    import pandas as pd

    import ibis.expr.datatypes as dt


_NATIVE_ARROW = True


class SnowflakeExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _has_reduction_filter_syntax = False
    _forbids_frame_clause = (
        *AlchemyExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
    )
    _require_order_by = (*AlchemyExprTranslator._require_order_by, ops.Reduction)
    _ignore_cast_types = (dt.Map, dt.Array)


class SnowflakeCompiler(AlchemyCompiler):
    translator_class = SnowflakeExprTranslator


class Backend(BaseAlchemyBackend):
    name = "snowflake"
    compiler = SnowflakeCompiler

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    @property
    def version(self) -> str:
        with self.begin() as con:
            return con.exec_driver_sql("SELECT CURRENT_VERSION()").scalar()

    def do_connect(
        self, user: str, password: str, account: str, database: str, **kwargs: Any
    ):
        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )
        url = URL(account=account, user=user, password=password, **dbparams, **kwargs)
        self.database_name = dbparams["database"]
        return super().do_connect(sa.create_engine(url))

    def _get_sqla_table(
        self, name: str, schema: str | None = None, **_: Any
    ) -> sa.Table:
        inspected = self.inspector.get_columns(name, schema)
        cols = []
        identifier = name if not schema else schema + "." + name
        with self.begin() as con:
            cur = con.execute(sa.text(f"DESCRIBE TABLE {identifier}")).mappings()
            for colname, colinfo in zip(toolz.pluck("name", cur), inspected):
                colinfo["name"] = colname
                colinfo["type_"] = colinfo.pop("type")
                cols.append(sa.Column(**colinfo, quote=True))
        return sa.Table(
            name,
            self.meta,
            *cols,
            schema=schema,
            extend_existing=True,
            keep_existing=False,
        )

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        global _NATIVE_ARROW

        if _NATIVE_ARROW:
            try:
                table = cursor.cursor.fetch_arrow_all()
            except sfc.NotSupportedError:
                _NATIVE_ARROW = False
            else:
                if table is None:
                    import pandas as pd

                    df = pd.DataFrame(columns=schema.names)
                else:
                    df = table.to_pandas(timestamp_as_object=True)
                return schema.apply_to(df)
        return super().fetch_from_cursor(cursor, schema)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        with self.begin() as bind:
            result = bind.execute(sa.text(f"SELECT * FROM ({query}) t0 LIMIT 0"))
            info_rows = bind.execute(
                sa.text(f"DESCRIBE RESULT {result.cursor.sfqid!r}")
            )

            for name, raw_type, null in toolz.pluck(
                ["name", "type", "null?"], info_rows.mappings()
            ):
                typ = parse(raw_type)
                yield name, typ(nullable=null.upper() == "Y")

    def list_databases(self, like=None) -> list[str]:
        with self.begin() as con:
            databases = [
                row.database_name
                for row in con.execute(
                    sa.text('SELECT database_name FROM information_schema.databases')
                ).mappings()
            ]
        return self._filter_with_like(databases, like)
