"""PostgreSQL backend."""

from __future__ import annotations

import contextlib

import sqlalchemy as sa

from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .compiler import PostgreSQLCompiler
from .udf import udf


class Backend(BaseAlchemyBackend):
    name = 'postgres'
    compiler = PostgreSQLCompiler

    def do_connect(
        self,
        host='localhost',
        user=None,
        password=None,
        port=5432,
        database=None,
        url=None,
        driver='psycopg2',
    ):
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
        url
            Complete SQLAlchemy connection string. If passed, the other
            connection arguments are ignored.
        driver
            Database driver

        Returns
        -------
        Backend
            Ibis backend instance

        Examples
        --------
        >>> import os
        >>> import getpass
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
            index : int64
            Unnamed: 0 : int64
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
            raise NotImplementedError(
                'psycopg2 is currently the only supported driver'
            )
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
        super().do_connect(sa.create_engine(alchemy_url))

    def list_databases(self, like=None):
        # http://dba.stackexchange.com/a/1304/58517
        databases = [
            row.datname
            for row in self.con.execute(
                'SELECT datname FROM pg_database WHERE NOT datistemplate'
            )
        ]
        return self._filter_with_like(databases, like)

    @util.deprecated(version='2.0', instead='`list_databases`')
    def list_schemas(self, like=None):
        """List all the schemas in the current database."""
        # In Postgres we support schemas, which in other engines (e.g. MySQL)
        # are databases
        return super().list_databases(like)

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute('SHOW TIMEZONE').scalar()
            bind.execute('SET TIMEZONE = UTC')
            try:
                yield bind
            finally:
                bind.execute(f"SET TIMEZONE = '{previous_timezone}'")

    def udf(
        self,
        pyfunc,
        in_types,
        out_type,
        schema=None,
        replace=False,
        name=None,
        language="plpythonu",
    ):
        """Decorator that defines a PL/Python UDF in-database.

        Parameters
        ----------
        pyfunc
            Python function
        in_types
            Input types
        out_type
            Output type
        schema
            The postgres schema in which to define the UDF
        replace
            replace UDF in database if already exists
        name
            name for the UDF to be defined in database
        language
            Language extension to use for PL/Python

        Returns
        -------
        Callable
            A callable ibis expression

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
            language=language,
        )
