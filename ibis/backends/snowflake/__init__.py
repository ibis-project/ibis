from __future__ import annotations

from typing import TYPE_CHECKING, Any

import snowflake.connector as sfc
import sqlalchemy as sa
from snowflake.sqlalchemy import URL

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


_NATIVE_PANDAS = True


class SnowflakeExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _has_reduction_filter_syntax = False


class SnowflakeCompiler(AlchemyCompiler):
    translator_class = SnowflakeExprTranslator


class Backend(BaseAlchemyBackend):
    name = "snowflake"
    compiler = SnowflakeCompiler

    def _convert_kwargs(self, kwargs):
        try:
            kwargs["account"] = kwargs.pop("host")
        except KeyError:
            pass

    @property
    def version(self) -> str:
        [(version,)] = self.con.execute("SELECT CURRENT_VERSION()").fetchall()
        return version

    def do_connect(
        self,
        user: str,
        password: str,
        account: str,
        database: str,
        **kwargs,
    ):
        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )
        url = URL(
            account=account,
            user=user,
            password=password,
            **dbparams,
            **kwargs,
        )
        self.database_name = dbparams["database"]
        return super().do_connect(sa.create_engine(url))

    def _get_sqla_table(
        self, name: str, schema: str | None = None, **_: Any
    ) -> sa.Table:
        inspected = self.inspector.get_columns(name)
        cols = []
        for (colname, *_), colinfo in zip(
            self.con.execute(f"DESCRIBE {name}"), inspected
        ):
            del colinfo["name"]
            colinfo["type_"] = colinfo.pop("type")
            cols.append(sa.Column(colname, **colinfo, quote=True))
        return sa.Table(
            name,
            self.meta,
            *cols,
            schema=schema,
            extend_existing=True,
            keep_existing=False,
        )

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        global _NATIVE_PANDAS

        if _NATIVE_PANDAS:
            try:
                df = cursor.cursor.fetch_pandas_all()
            except sfc.NotSupportedError:
                _NATIVE_PANDAS = False
            else:
                return schema.apply_to(df)
        return super().fetch_from_cursor(cursor, schema)

    def _get_schema_using_query(self, query):
        with self.begin() as bind:
            result = bind.execute(f"SELECT * FROM ({query}) t0 LIMIT 0")
            info_rows = bind.execute(f"DESCRIBE RESULT {result.cursor.sfqid!r}")

        schema = {}
        for name, raw_type, _, null, *_ in info_rows:
            typ = parse(raw_type)
            schema[name] = typ(nullable=null.upper() == "Y")
        return sch.Schema.from_dict(schema)
