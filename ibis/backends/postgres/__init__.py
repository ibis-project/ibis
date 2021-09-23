"""PostgreSQL backend."""
from ibis.backends.base import BaseBackend
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .client import PostgreSQLClient


class Backend(BaseAlchemyBackend, BaseBackend):
    name = 'postgres'
    client_class = PostgreSQLClient

    def connect(
        self,
        host='localhost',
        user=None,
        password=None,
        port=5432,
        database=None,
        url=None,
        driver='psycopg2',
    ):
        """Create an Ibis client located at `user`:`password`@`host`:`port`
        connected to a PostgreSQL database named `database`.

        Parameters
        ----------
        host : string, default 'localhost'
        user : string, default None
        password : string, default None
        port : string or integer, default 5432
        database : string, default None
        url : string, default None
            Complete SQLAlchemy connection string. If passed, the other
            connection arguments are ignored.
        driver : string, default 'psycopg2'

        Returns
        -------
        PostgreSQLClient

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
        self.client = PostgreSQLClient(
            backend=self,
            host=host,
            user=user,
            password=password,
            port=port,
            database=database,
            url=url,
            driver=driver,
        )
        return self.client

    def list_databases(self, like=None):
        # http://dba.stackexchange.com/a/1304/58517
        databases = [
            row.datname
            for row in self.client.con.execute(
                'SELECT datname FROM pg_database WHERE NOT datistemplate'
            )
        ]
        return self._filter_with_like(databases, like)

    def list_schemas(self, like=None):
        """List all the schemas in the current database."""
        # In Postgres we support schemas, which in other engines (e.g. MySQL)
        # are databases
        return super().list_databases(like)

    def list_tables(self, like=None, database=None):
        # PostgreSQL requires passing `database=None` for current database
        if database == self.current_database:
            database = None
        return super().list_tables(like, database)
