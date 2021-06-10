"""PostgreSQL backend."""
from ibis.backends.base import BaseBackend

from .client import PostgreSQLClient, PostgreSQLDatabase, PostgreSQLTable


class Backend(BaseBackend):
    name = 'postgres'
    kind = 'sqlalchemy'
    client = PostgreSQLClient
    database_class = PostgreSQLDatabase
    table_class = PostgreSQLTable

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
        return PostgreSQLClient(
            backend=self,
            host=host,
            user=user,
            password=password,
            port=port,
            database=database,
            url=url,
            driver=driver,
        )
