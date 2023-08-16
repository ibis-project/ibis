from __future__ import annotations

import re
import warnings
from collections import ChainMap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
import sqlglot as sg

from ibis import util
from ibis.backends.base.sql.alchemy import AlchemyCanCreateSchema, BaseAlchemyBackend
from ibis.backends.base.sqlglot.datatypes import PostgresType
from ibis.backends.exasol.compiler import ExasolCompiler

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

    from ibis.backends.base import BaseBackend
    from ibis.expr import datatypes as dt


class Backend(BaseAlchemyBackend, AlchemyCanCreateSchema):
    name = "exasol"
    compiler = ExasolCompiler
    supports_temporary_tables = False
    supports_create_or_replace = False
    supports_in_memory_tables = False
    supports_python_udfs = False

    def do_connect(
        self,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 8563,
        schema: str | None = None,
        encryption: bool = True,
        certificate_validation: bool = True,
        encoding: str = "en_US.UTF-8",
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
        port
            Port number to connect to (default: 8563)
        schema
            Database schema to open, if `None`, no schema will be opened.
        encryption
            Enables/disables transport layer encryption (default: True).
        certificate_validation
            Enables/disables certificate validation (default: True).
        encoding
            The encoding format (default: "en_US.UTF-8").
        """
        options = [
            "SSLCertificate=SSL_VERIFY_NONE" if not certificate_validation else "",
            f"ENCRYPTION={'yes' if encryption else 'no'}",
            f"CONNECTIONCALL={encoding}",
        ]
        url_template = (
            "exa+websocket://{user}:{password}@{host}:{port}/{schema}?{options}"
        )
        url = sa.engine.url.make_url(
            url_template.format(
                user=user,
                password=password,
                host=host,
                port=port,
                schema=schema,
                options="&".join(options),
            )
        )
        engine = sa.create_engine(url, poolclass=sa.pool.StaticPool)
        super().do_connect(engine)

    def _convert_kwargs(self, kwargs: MutableMapping) -> None:
        def convert_sqla_to_ibis(keyword_arguments):
            sqla_to_ibis = {"tls": "encryption", "username": "user"}
            for sqla_kwarg, ibis_kwarg in sqla_to_ibis.items():
                if sqla_kwarg in keyword_arguments:
                    keyword_arguments[ibis_kwarg] = keyword_arguments.pop(sqla_kwarg)

        def filter_kwargs(keyword_arguments):
            allowed_parameters = [
                "user",
                "password",
                "host",
                "port",
                "schema",
                "encryption",
                "certificate",
                "encoding",
            ]
            to_be_removed = [
                key for key in keyword_arguments if key not in allowed_parameters
            ]
            for parameter_name in to_be_removed:
                del keyword_arguments[parameter_name]

        convert_sqla_to_ibis(kwargs)
        filter_kwargs(kwargs)

    def _from_url(self, url: str, **kwargs) -> BaseBackend:
        """Construct an ibis backend from a SQLAlchemy-conforming URL."""
        kwargs = ChainMap(kwargs)
        _, new_kwargs = self.inspector.dialect.create_connect_args(url)
        kwargs = kwargs.new_child(new_kwargs)
        kwargs = dict(kwargs)
        self._convert_kwargs(kwargs)

        return self.connect(**kwargs)

    @property
    def inspector(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sa.exc.RemovedIn20Warning)
            return super().inspector

    @contextmanager
    def begin(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sa.exc.RemovedIn20Warning)
            with super().begin() as con:
                yield con

    def list_tables(self, like=None, database=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sa.exc.RemovedIn20Warning)
            return super().list_tables(like=like, database=database)

    def _get_sqla_table(
        self,
        name: str,
        autoload: bool = True,
        **kwargs: Any,
    ) -> sa.Table:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sa.exc.RemovedIn20Warning)
            return super()._get_sqla_table(name=name, autoload=autoload, **kwargs)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        table = sg.table(util.gen_name("exasol_metadata"))
        create_view = sg.exp.Create(
            kind="VIEW", this=table, expression=sg.parse_one(query, dialect="postgres")
        )
        drop_view = sg.exp.Drop(kind="VIEW", this=table)
        describe = sg.exp.Describe(this=table).sql(dialect="postgres")
        # strip trailing encodings e.g., UTF8
        varchar_regex = re.compile(r"^(VARCHAR(?:\(\d+\)))?(?:\s+.+)?$")
        with self.begin() as con:
            con.exec_driver_sql(create_view.sql(dialect="postgres"))
            try:
                yield from (
                    (
                        name,
                        PostgresType.from_string(varchar_regex.sub(r"\1", typ)),
                    )
                    for name, typ, *_ in con.exec_driver_sql(describe)
                )
            finally:
                con.exec_driver_sql(drop_view.sql(dialect="postgres"))

    @property
    def current_schema(self) -> str:
        return self._scalar_query(sa.select(sa.text("CURRENT_SCHEMA")))

    @property
    def current_database(self) -> str:
        return None

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )
        drop_schema = sg.exp.Drop(
            kind="SCHEMA", this=sg.to_identifier(name), exists=force
        )
        with self.begin() as con:
            con.exec_driver_sql(drop_schema.sql(dialect="postgres"))

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )
        create_schema = sg.exp.Create(
            kind="SCHEMA", this=sg.to_identifier(name), exists=force
        )
        with self.begin() as con:
            open_schema = self.current_schema
            con.exec_driver_sql(create_schema.sql(dialect="postgres"))
            # Exasol implicitly opens the created schema, therefore we need to restore
            # the previous context.
            action = (
                sa.text(f"OPEN SCHEMA {open_schema}")
                if open_schema
                else sa.text(f"CLOSE SCHEMA {name}")
            )
            con.exec_driver_sql(action)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        if database is not None:
            raise NotImplementedError(
                "`database` argument is not supported for the Exasol backend"
            )

        schema, table = "SYS", "EXA_SCHEMAS"
        sch = sa.table(
            table,
            sa.column("schema_name", sa.TEXT()),
            schema=schema,
        )

        query = sa.select(sch.c.schema_name)

        with self.begin() as con:
            schemas = list(con.execute(query).scalars())
        return self._filter_with_like(schemas, like=like)
