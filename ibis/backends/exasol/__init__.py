from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import pyexasol
import sqlglot as sg

import ibis.common.exceptions as com
from ibis import util
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.exasol.compiler import ExasolCompiler

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import pyarrow as pa

    import ibis.expr.datatypes as dt
    import ibis.expr.schema as sch
    import ibis.expr.types as ir
    from ibis.backends.base import BaseBackend


class Backend(SQLGlotBackend):
    name = "exasol"
    compiler = ExasolCompiler
    supports_temporary_tables = False
    supports_create_or_replace = False
    supports_in_memory_tables = False
    supports_python_udfs = False

    @property
    def version(self) -> str:
        with self._safe_raw_sql("SELECT version()") as result:
            [(version,)] = result.fetchall()
        return version

    def do_connect(
        self,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 8563,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to an Exasol database.

        Parameters
        ----------
        user
            Username used for authentication.
        password
            Password used for authentication.
        host
            Hostname to connect to (default: "localhost").
            **kwargs
        port
            Port number to connect to (default: 8563)
        kwargs
            Additional keyword arguments passed to `pyexasol.connect`.
        """
        self.con = pyexasol.connect(
            dsn=f"{host}:{port}", user=user, password=password, **kwargs
        )
        self._temp_views = set()

    def _from_url(self, url: str, **kwargs) -> BaseBackend:
        """Construct an ibis backend from a SQLAlchemy-conforming URL."""
        url = urlparse(url)
        query_params = parse_qs(url.query)
        kwargs = {
            "user": url.username,
            "password": url.password,
            "schema": url.path[1:] or None,
            "host": url.hostname,
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
    def _safe_raw_sql(self, query: str, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.compiler.dialect)

        with self.begin() as cur:
            cur.execute(query, *args, **kwargs)
            yield cur

    def list_tables(self, like=None, database=None):
        return super().list_tables(like=like, database=database)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        table = sg.table(util.gen_name("exasol_metadata"))
        create_view = sg.exp.Create(
            kind="VIEW",
            this=table,
            expression=sg.parse_one(query, dialect=self.compiler.dialect),
        )
        drop_view = sg.exp.Drop(kind="VIEW", this=table)
        describe = sg.exp.Describe(this=table)
        # strip trailing encodings e.g., UTF8
        varchar_regex = re.compile(r"^(VARCHAR(?:\(\d+\)))?(?:\s+.+)?$")
        with self._safe_raw_sql(create_view) as con:
            try:
                con.execute(describe.sql(dialect=self.compiler.dialect))
                yield from (
                    (
                        name,
                        self.compiler.type_mapper.from_string(
                            varchar_regex.sub(r"\1", typ)
                        ),
                    )
                    for name, typ, *_ in con.fetchall()
                )
            finally:
                con.execute(drop_view.sql(dialect=self.compiler.dialect))

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

    @property
    def current_schema(self) -> str:
        with self._safe_raw_sql("SELECT CURRENT_SCHEMA") as cur:
            [(schema,)] = cur.fetchall()
        return schema

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )
        drop_schema = sg.exp.Drop(kind="SCHEMA", this=name, exists=force)
        with self.begin() as con:
            con.execute(drop_schema.sql(dialect=self.compiler.dialect))

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )
        create_schema = sg.exp.Create(kind="SCHEMA", this=name, exists=force)
        open_schema = self.current_schema
        with self.begin() as con:
            con.execute(create_schema.sql(dialect=self.compiler.dialect))
            # Exasol implicitly opens the created schema, therefore we need to restore
            # the previous context.
            con.execute(
                f"OPEN SCHEMA {open_schema}" if open_schema else f"CLOSE SCHEMA {name}"
            )

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )

        query = sg.select("schema_name").from_(sg.table("EXA_SCHEMAS", catalog="SYS"))

        with self._safe_raw_sql(query) as con:
            schemas = con.fetchall()
        return self._filter_with_like([schema for (schema,) in schemas], like=like)
