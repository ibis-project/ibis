from __future__ import annotations

import contextlib
import io
import json
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.json
import requests
from pydantic import Field

import ibis
import ibis.config
import ibis.expr.schema as sch
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.clickhouse.client import ClickhouseTable, fully_qualified_re
from ibis.backends.clickhouse.compiler import ClickhouseCompiler
from ibis.backends.clickhouse.datatypes import parse, serialize
from ibis.backends.pyarrow.datatypes import to_pyarrow_type
from ibis.config import options


def _dataframe_to_arrow_bytes(df):
    t = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, schema := t.schema) as writer:
        writer.write_table(t)
    return buf.getvalue(), schema


class Backend(BaseSQLBackend):
    name = 'clickhouse'
    table_expr_class = ClickhouseTable
    compiler = ClickhouseCompiler

    class Options(ibis.config.BaseModel):
        temp_db: str = Field(
            default="__ibis_tmp",
            description="Database to use for temporary objects.",
        )

    def do_connect(
        self,
        host: str = "localhost",
        port: int | None = None,
        database: str = "default",
        user: str = "default",
        password: str = "",
        secure: bool = False,
        **kwargs: Any,
    ):
        """Create a ClickHouse client for use with Ibis.

        Parameters
        ----------
        host
            Host name of the clickhouse server
        port
            Clickhouse server's  port
        database
            Default database when executing queries
        user
            User to authenticate with
        password
            Password to authenticate with

        Examples
        --------
        >>> import ibis
        >>> import os
        >>> clickhouse_host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        >>> clickhouse_port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        >>> client = ibis.clickhouse.connect(host=clickhouse_host,  port=clickhouse_port)
        >>> client  # doctest: +ELLIPSIS
        <ibis.clickhouse.client.ClickhouseClient object at 0x...>
        """  # noqa: E501
        self.session = requests.Session()
        self.session.auth = (user, password)
        self.host = host
        self.database = database
        self.params = kwargs
        self.secure = secure
        self.port = 8123 if not secure and port is None else port

    @contextlib.contextmanager
    def _database(self, database: str | None):
        if database is not None:
            prev_database = self.database
            self.database = database
            try:
                yield
            finally:
                self.database = prev_database
        else:
            yield

    @property
    def url(self) -> str:
        scheme = f"http{'s' * self.secure}"
        port = self.port
        netloc = f"{self.host}" + f":{port}" * (port is not None)
        db = self.database
        path = f"/{db}" * bool(db)
        return f"{scheme}://{netloc}{path}"

    def _get(
        self,
        query: str,
        *,
        format: str = "JSONColumnsWithMetadata",
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if params is None:
            params = {}
        with self.session as sesh:
            resp = sesh.get(
                self.url,
                params={
                    **self.params,
                    **params,
                    **dict(query=query, default_format=format),
                },
            )
        resp.raise_for_status()
        return resp

    def _post(
        self,
        query: str,
        data: bytes | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        files: Mapping | None = None,
    ) -> dict[str, Any]:
        if params is None:
            params = {}

        with self.session as sesh:
            resp = sesh.post(
                self.url,
                data=data,
                params={
                    **self.params,
                    **params,
                    **dict(query=query),
                },
                files=files,
            )
        resp.raise_for_status()
        return resp

    @property
    def version(self) -> str:
        resp = self._get("SELECT VERSION() AS v")
        js = resp.json()
        [version] = js["data"]["v"]
        return version

    @property
    def current_database(self):
        return self.database

    def list_databases(self, like=None):
        data = self.raw_sql("SELECT name FROM system.databases")
        return self._filter_with_like(data["name"].to_pylist(), like)

    def list_tables(self, like=None, database=None):
        with self._database(database):
            data = self.raw_sql("SHOW TABLES")
        return self._filter_with_like(data["name"].to_pylist(), like)

    @staticmethod
    def _construct_external_tables(external_tables):
        files = {}
        data = {}
        for name, df in external_tables.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    "External table is not an instance of pandas dataframe"
                )
            schema = sch.infer(df)

            files[name] = df.to_json(
                index=False,
                orient="records",
                lines=True,
                date_format="epoch",
                date_unit="ns",
            )
            data[f"{name}_format"] = "JSONEachRow"
            data[f"{name}_types"] = ",".join(
                serialize(typ) for typ in schema.types
            )

        return files, data

    def raw_sql(
        self,
        query: str,
        external_tables: Mapping[str, pd.DataFrame] | None = None,
    ) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        external_tables
            Mapping of table name to pandas DataFrames providing
            external datasources for the query

        Returns
        -------
        Any
            The resutls of executing the query
        """
        if external_tables is None:
            external_tables = {}

        params = dict(
            # tell clickhouse to return arrow strings as strings instead of
            # binary
            output_format_arrow_string_as_string=1,
            # map low cardinality columns to dictionary encoded columns
            output_format_arrow_low_cardinality_as_dictionary=1,
            # use arrow stream as the output format
            default_format="JSONEachRow",
        )
        files, data = self._construct_external_tables(external_tables)
        ibis.util.log(query)
        resp = self._post(query, data=data, params=params, files=files)

        if content := resp.content:
            t = pa.json.read_json(io.BytesIO(content))
            return t
        else:
            breakpoint()
            ...

    def fetch_from_cursor(self, cursor, schema):
        df = cursor.to_pandas()
        breakpoint()
        return schema.apply_to(df)

    def close(self):
        """Close Clickhouse connection and drop any temporary objects"""
        self.session.close()

    def _fully_qualified_name(self, name, database):
        return name

    def get_schema(
        self,
        table_name: str,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May be fully qualified
        database
            Database name

        Returns
        -------
        ibis.expr.schema.Schema
            Ibis schema
        """
        with self._database(database):
            data = self.raw_sql(f"DESCRIBE {table_name}")

        return sch.Schema.from_tuples(
            zip(data["name"].to_pylist(), map(parse, data["type"].to_pylist()))
        )

    def set_options(self, options):
        raise NotImplementedError("set_options not implemented for ClickHouse")

    def reset_options(self):
        raise NotImplementedError(
            "reset_options not implemented for ClickHouse"
        )

    def _ensure_temp_db_exists(self):
        name = (options.clickhouse.temp_db,)
        if name not in self.list_databases():
            self.create_database(name, force=True)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        plans = self.raw_sql(
            f"EXPLAIN json = 1, description = 0, header = 1 {query}"
        )
        [raw_plan] = plans["explain"]
        [plan] = json.loads(raw_plan.as_py())
        fields = [
            (field["Name"], parse(field["Type"]))
            for field in plan["Plan"]["Header"]
        ]
        return sch.Schema.from_tuples(fields)
