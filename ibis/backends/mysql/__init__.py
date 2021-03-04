from ibis.backends.base import BaseBackend

from ibis.backends.base_sqlalchemy.alchemy import (
    AlchemyQueryBuilder,
)

from .client import MySQLClient
from .compiler import MySQLDialect


def connect(
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
        Complete SQLAlchemy connection string. If passed, the other connection
        arguments are ignored.
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
    return MySQLClient(
        host=host,
        user=user,
        password=password,
        port=port,
        database=database,
        url=url,
        driver=driver,
    )


class Backend(BaseBackend):
    name = 'mysql'
    builder = AlchemyQueryBuilder
    dialect = MySQLDialect
    connect = connect
