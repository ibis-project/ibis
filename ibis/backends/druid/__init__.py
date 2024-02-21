"""The Druid backend."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import pydruid
import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.druid.compiler import DruidCompiler
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compiler import STAR
from ibis.backends.sql.datatypes import DruidType

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd
    import pyarrow as pa

    import ibis.expr.types as ir


class Backend(SQLBackend):
    name = "druid"
    compiler = DruidCompiler()
    supports_create_or_replace = False
    supports_in_memory_tables = True

    @property
    def version(self) -> str:
        with self._safe_raw_sql("SELECT version()") as result:
            [(version,)] = result.fetchall()
        return version

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
        query_params = parse_qs(url.query)
        kwargs = {
            "user": url.username,
            "password": url.password,
            "host": url.hostname,
            "path": url.path,
            "port": url.port,
        } | kwargs

        for name, value in query_params.items():
            if len(value) > 1:
                kwargs[name] = value
            elif len(value) == 1:
                kwargs[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        self._convert_kwargs(kwargs)

        return self.connect(**kwargs)

    @property
    def current_database(self) -> str:
        # https://druid.apache.org/docs/latest/querying/sql-metadata-tables.html#schemata-table
        return "druid"

    def do_connect(self, **kwargs: Any) -> None:
        """Create an Ibis client using the passed connection parameters."""
        header = kwargs.pop("header", True)
        self.con = pydruid.db.connect(**kwargs, header=header)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        with contextlib.closing(self.con.cursor()) as cur:
            cur.execute(query, *args, **kwargs)
            yield cur

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        with self._safe_raw_sql(f"EXPLAIN PLAN FOR {query}") as result:
            [(row, *_)] = result.fetchall()

        (plan,) = json.loads(row)
        for column in plan["signature"]:
            name, typ = column["name"], column["type"]
            if name == "__time":
                dtype = dt.timestamp
            else:
                dtype = DruidType.from_string(typ)
            yield name, dtype

    def get_schema(
        self, table_name: str, schema: str | None = None, database: str | None = None
    ) -> sch.Schema:
        name_type_pairs = self._metadata(
            sg.select(STAR)
            .from_(sg.table(table_name, db=schema, catalog=database))
            .sql(self.dialect)
        )
        return sch.Schema.from_tuples(name_type_pairs)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.formats.pandas import PandasData

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            cursor.close()
            raise
        df = PandasData.convert_table(df, schema)
        return df

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        raise NotImplementedError()

    def drop_table(self, *args, **kwargs):
        raise NotImplementedError()

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        t = sg.table("TABLES", db="INFORMATION_SCHEMA", quoted=True)
        c = self.compiler
        query = sg.select(sg.column("TABLE_NAME", quoted=True)).from_(t).sql(c.dialect)

        with self._safe_raw_sql(query) as result:
            tables = result.fetchall()
        return self._filter_with_like([table.TABLE_NAME for table in tables], like=like)

    def _register_in_memory_tables(self, expr):
        """No-op. Table are inlined, for better or worse."""

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)

        dtypes = expr.as_table().schema().values()

        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            while batch := cursor.fetchmany(chunk_size):
                yield (tuple(map(dt.normalize, dtypes, row)) for row in batch)
