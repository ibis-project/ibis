"""The Druid backend."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import pydruid.db
import sqlglot as sg
import sqlglot.expressions as sge

import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR
from ibis.backends.sql.datatypes import DruidType
from ibis.backends.tests.errors import PyDruidProgrammingError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import pyarrow as pa

    import ibis.expr.operations as ops
    import ibis.expr.types as ir


class Backend(SQLBackend):
    name = "druid"
    compiler = sc.druid.compiler
    supports_create_or_replace = False

    @property
    def version(self) -> str:
        with self._safe_raw_sql("SELECT version()") as result:
            [(version,)] = result.fetchall()
        return version

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
        kwargs = {
            "user": url.username,
            "password": unquote_plus(url.password)
            if url.password is not None
            else None,
            "host": url.hostname,
            "path": url.path,
            "port": url.port,
            **kwargs,
        }

        self._convert_kwargs(kwargs)

        return self.connect(**kwargs)

    @property
    def current_database(self) -> str:
        # https://druid.apache.org/docs/latest/querying/sql-metadata-tables.html#schemata-table
        return "druid"

    def do_connect(self, **kwargs: Any) -> None:
        """Create an Ibis client using the passed connection parameters.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.connect("druid://localhost:8082/druid/v2/sql?header=true")
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          __time          timestamp
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
          timestamp_col   int64
          year            int64
          month           int64
        """
        header = kwargs.pop("header", True)
        self.con = pydruid.db.connect(**kwargs, header=header)

    @util.experimental
    @classmethod
    def from_connection(cls, con: pydruid.db.api.Connection) -> Backend:
        """Create an Ibis client from an existing connection to a Druid database.

        Parameters
        ----------
        con
            An existing connection to a Druid database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        return new_backend

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        with contextlib.closing(self.con.cursor()) as cur:
            cur.execute(query, *args, **kwargs)
            yield cur

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with self._safe_raw_sql(f"EXPLAIN PLAN FOR {query}") as result:
            [(row, *_)] = result.fetchall()

        (plan,) = json.loads(row)

        schema = {}

        for column in plan["signature"]:
            name, typ = column["name"], column["type"]
            if name == "__time":
                dtype = dt.timestamp
            else:
                dtype = DruidType.from_string(typ)
            schema[name] = dtype
        return sch.Schema(schema)

    def _table_exists(self, name: str):
        quoted = self.compiler.quoted
        t = sg.table("TABLES", db="INFORMATION_SCHEMA", quoted=quoted)
        table_name = sg.column("TABLE_NAME", quoted=quoted)
        query = (
            sg.select(table_name)
            .from_(t)
            .where(table_name.eq(sge.convert(name)))
            .sql(self.dialect)
        )

        with self._safe_raw_sql(query) as result:
            tables = result.fetchall()
        return bool(tables)

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        query = (
            sg.select(STAR)
            .from_(sg.table(table_name, db=database, catalog=catalog))
            .sql(self.dialect)
        )
        try:
            schema = self._get_schema_using_query(query)
        except PyDruidProgrammingError as e:
            if not self._table_exists(table_name):
                raise com.TableNotFound(table_name) from e
            raise

        return schema

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
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database.
        """
        t = sg.table("TABLES", db="INFORMATION_SCHEMA", quoted=True)
        c = self.compiler
        query = sg.select(sg.column("TABLE_NAME", quoted=True)).from_(t).sql(c.dialect)

        with self._safe_raw_sql(query) as result:
            tables = result.fetchall()
        return self._filter_with_like([table.TABLE_NAME for table in tables], like=like)

    def _register_in_memory_table(self, op: ops.InMemoryTable):
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
