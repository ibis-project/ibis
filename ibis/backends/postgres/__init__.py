"""PostgreSQL backend."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING, Callable, Iterable, Literal

import sqlalchemy as sa

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.postgres.compiler import PostgreSQLCompiler
from ibis.backends.postgres.datatypes import _get_type
from ibis.common.exceptions import InvalidDecoratorError

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


def _verify_source_line(func_name: str, line: str):
    if line.startswith("@"):
        raise InvalidDecoratorError(func_name, line)
    return line


class Backend(BaseAlchemyBackend):
    name = "postgres"
    compiler = PostgreSQLCompiler
    supports_create_or_replace = False
    supports_python_udfs = True

    def do_connect(
        self,
        host: str = 'localhost',
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
        url: str | None = None,
        driver: Literal["psycopg2"] = "psycopg2",
    ) -> None:
        """Create an Ibis client connected to PostgreSQL database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number
        database
            Database to connect to
        schema
            PostgreSQL schema to use. If `None`, use the default `search_path`.
        url
            SQLAlchemy connection string.

            If passed, the other connection arguments are ignored.
        driver
            Database driver

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> import ibis
        >>> host = os.environ.get('IBIS_TEST_POSTGRES_HOST', 'localhost')
        >>> user = os.environ.get('IBIS_TEST_POSTGRES_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
        ...                           'ibis_testing')
        >>> con = connect(
        ...     database=database,
        ...     host=host,
        ...     user=user,
        ...     password=password
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table('functional_alltypes')
        >>> t
        PostgreSQLTable[table]
          name: functional_alltypes
          schema:
            id : int32
            bool_col : boolean
            tinyint_col : int16
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32
        """
        if driver != 'psycopg2':
            raise NotImplementedError('psycopg2 is currently the only supported driver')

        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f'postgresql+{driver}',
        )
        self.database_name = alchemy_url.database

        connect_args = {}
        if schema is not None:
            connect_args["options"] = f"-csearch_path={schema}"

        engine = sa.create_engine(
            alchemy_url, connect_args=connect_args, poolclass=sa.pool.StaticPool
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                cur.execute("SET TIMEZONE = UTC")

        super().do_connect(engine)

    def list_databases(self, like=None):
        with self.begin() as con:
            # http://dba.stackexchange.com/a/1304/58517
            databases = [
                row.datname
                for row in con.exec_driver_sql(
                    "SELECT datname FROM pg_database WHERE NOT datistemplate"
                ).mappings()
            ]
        return self._filter_with_like(databases, like)

    def function(self, name: str, *, schema: str | None = None) -> Callable:
        query = sa.text(
            """
SELECT
  n.nspname as schema,
  pg_catalog.pg_get_function_result(p.oid) as return_type,
  string_to_array(pg_catalog.pg_get_function_arguments(p.oid), ', ') as signature,
  CASE p.prokind
    WHEN 'a' THEN 'agg'
    WHEN 'w' THEN 'window'
    WHEN 'p' THEN 'proc'
    ELSE 'func'
  END as "Type"
FROM pg_catalog.pg_proc p
LEFT JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = :name
"""
            + "AND n.nspname OPERATOR(pg_catalog.~) :schema COLLATE pg_catalog.default"
            * (schema is not None)
        ).bindparams(name=name, schema=f"^({schema})$")

        def split_name_type(arg: str) -> tuple[str, dt.DataType]:
            name, typ = arg.split(" ", 1)
            return name, _get_type(typ)

        with self.begin() as con:
            rows = con.execute(query).mappings().fetchall()

            if not rows:
                name = f"{schema}.{name}" if schema else name
                raise exc.MissingUDFError(name)
            elif len(rows) > 1:
                raise exc.AmbiguousUDFError(name)

            [row] = rows
            return_type = _get_type(row["return_type"])
            signature = list(map(split_name_type, row["signature"]))

        # dummy callable
        def fake_func(*args, **kwargs):
            ...

        fake_func.__name__ = name
        fake_func.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ
                )
                for name, typ in signature
            ],
            return_annotation=return_type,
        )
        fake_func.__annotations__ = {"return": return_type, **dict(signature)}
        op = ops.udf.scalar._opaque(fake_func, schema=schema)
        return op

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        config = udf_node.__config__["kwargs"]
        func = udf_node.__func__
        func_name = func.__name__
        schema = config.get("schema", "")
        name = type(udf_node).__name__
        ident = ".".join(filter(None, [schema, name]))
        return dict(
            name=name,
            ident=ident,
            signature=", ".join(
                f"{name} {self._compile_type(arg.output_dtype)}"
                for name, arg in zip(udf_node.argnames, udf_node.args)
            ),
            return_type=self._compile_type(udf_node.output_dtype),
            language=config.get("language", "plpython3u"),
            source="\n".join(
                _verify_source_line(func_name, line)
                for line in textwrap.dedent(inspect.getsource(func)).splitlines()
                if not line.strip().startswith("@udf")
            ),
            args=", ".join(udf_node.argnames),
        )

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
CREATE OR REPLACE FUNCTION {ident}({signature})
RETURNS {return_type}
LANGUAGE {language}
AS $$
{source}
return {name}({args})
$$""".format(
            **self._get_udf_source(udf_node)
        )

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        name = util.gen_name("postgres_metadata")
        type_info_sql = """\
SELECT
  attname,
  format_type(atttypid, atttypmod) AS type
FROM pg_attribute
WHERE attrelid = CAST(:name AS regclass)
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum"""
        if self.inspector.has_table(query):
            query = f"TABLE {query}"

        text = sa.text(type_info_sql).bindparams(name=name)
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE TEMPORARY VIEW {name} AS {query}")
            try:
                yield from (
                    (col, _get_type(typestr)) for col, typestr in con.execute(text)
                )
            finally:
                con.exec_driver_sql(f"DROP VIEW IF EXISTS {name}")

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"DROP VIEW IF EXISTS {name}"
        yield f"CREATE TEMPORARY VIEW {name} AS {definition}"
