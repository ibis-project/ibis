import contextlib
import warnings

from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .client import MySQLClient
from .compiler import MySQLCompiler


class Backend(BaseAlchemyBackend):
    name = 'mysql'
    client_class = MySQLClient
    compiler = MySQLCompiler

    def connect(
        self,
        host='localhost',
        user=None,
        password=None,
        port=3306,
        database=None,
        url=None,
        driver='pymysql',
    ):

        """Create an Ibis client located at `user`:`password`@`host`:`port`
        connected to a MySQL database named `database`.

        Parameters
        ----------
        host : string, default 'localhost'
        user : string, default None
        password : string, default None
        port : string or integer, default 3306
        database : string, default None
        url : string, default None
            Complete SQLAlchemy connection string. If passed, the other
            connection arguments are ignored.
        driver : string, default 'pymysql'

        Returns
        -------
        MySQLClient

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> host = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
        >>> user = os.environ.get('IBIS_TEST_MYSQL_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_MYSQL_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_MYSQL_DATABASE',
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
        MySQLTable[table]
          name: functional_alltypes
          schema:
            index : int64
            Unnamed: 0 : int64
            id : int32
            bool_col : int8
            tinyint_col : int8
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
        self.client = MySQLClient(
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

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute(
                'SELECT @@session.time_zone'
            ).scalar()
            try:
                bind.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Couldn't set mysql timezone: {str(e)}")

            try:
                yield bind
            finally:
                query = "SET @@session.time_zone = '{}'"
                bind.execute(query.format(previous_timezone))

    def table(self, name, database=None, schema=None):
        """Create a table expression that references a particular a table
        called `name` in a MySQL database called `database`.

        Parameters
        ----------
        name : str
            The name of the table to retrieve.
        database : str, optional
            The database in which the table referred to by `name` resides. If
            ``None`` then the ``current_database`` is used.
        schema : str, optional
            The schema in which the table resides.  If ``None`` then the
            `public` schema is assumed.

        Returns
        -------
        table : TableExpr
            A table expression.
        """
        if database is not None and database != self.current_database:
            return self.database(name=database).table(name=name, schema=schema)
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = self.table_class(alch_table, self, self._schemas.get(name))
            return self.table_expr_class(node)
