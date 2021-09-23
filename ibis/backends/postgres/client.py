import contextlib
import getpass
from typing import Optional

import psycopg2  # NOQA fail early if the driver is missing
import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import AlchemyClient
from ibis.backends.postgres import udf

from .compiler import PostgreSQLCompiler


class PostgreSQLClient(AlchemyClient):

    """The Ibis PostgreSQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    compiler = PostgreSQLCompiler

    def __init__(
        self,
        backend,
        host: str = 'localhost',
        user: str = getpass.getuser(),
        password: Optional[str] = None,
        port: int = 5432,
        database: str = 'public',
        url: Optional[str] = None,
        driver: str = 'psycopg2',
    ):
        self.backend = backend
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        if url is None:
            if driver != 'psycopg2':
                raise NotImplementedError(
                    'psycopg2 is currently the only supported driver'
                )
            sa_url = sa.engine.url.URL(
                'postgresql+psycopg2',
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
            )
        else:
            sa_url = sa.engine.url.make_url(url)

        super().__init__(sa.create_engine(sa_url))
        self.database_name = sa_url.database

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute('SHOW TIMEZONE').scalar()
            bind.execute('SET TIMEZONE = UTC')
            try:
                yield bind
            finally:
                bind.execute(f"SET TIMEZONE = '{previous_timezone}'")

    def list_schemas(self, like=None):
        return self.backend.list_schemas(like)

    def udf(
        self, pyfunc, in_types, out_type, schema=None, replace=False, name=None
    ):
        """Decorator that defines a PL/Python UDF in-database based on the
        wrapped function and turns it into an ibis function expression.

        Parameters
        ----------
        pyfunc : function
        in_types : List[ibis.expr.datatypes.DataType]
        out_type : ibis.expr.datatypes.DataType
        schema : str
            optionally specify the schema in which to define the UDF
        replace : bool
            replace UDF in database if already exists
        name: str
            name for the UDF to be defined in database

        Returns
        -------
        Callable

        Function that takes in ColumnExpr arguments and returns an instance
        inheriting from PostgresUDFNode
        """

        return udf(
            client=self,
            python_func=pyfunc,
            in_types=in_types,
            out_type=out_type,
            schema=schema,
            replace=replace,
            name=name,
        )
