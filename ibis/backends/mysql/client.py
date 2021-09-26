import getpass

import pymysql  # NOQA fail early if the driver is missing
import sqlalchemy as sa
import sqlalchemy.dialects.mysql as mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import AlchemyClient

# TODO(kszucs): unsigned integers


@dt.dtype.register((mysql.DOUBLE, mysql.REAL))
def mysql_double(satype, nullable=True):
    return dt.Double(nullable=nullable)


@dt.dtype.register(mysql.FLOAT)
def mysql_float(satype, nullable=True):
    return dt.Float(nullable=nullable)


@dt.dtype.register(mysql.TINYINT)
def mysql_tinyint(satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(mysql.BLOB)
def mysql_blob(satype, nullable=True):
    return dt.Binary(nullable=nullable)


class MySQLClient(AlchemyClient):

    """The Ibis MySQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    def __init__(
        self,
        backend,
        host='localhost',
        user=None,
        password=None,
        port=3306,
        database='mysql',
        url=None,
        driver='pymysql',
    ):
        self.backend = backend
        if url is None:
            if driver != 'pymysql':
                raise NotImplementedError(
                    'pymysql is currently the only supported driver'
                )
            user = user or getpass.getuser()
            url = sa.engine.url.URL(
                'mysql+pymysql',
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
            )
        else:
            url = sa.engine.url.make_url(url)

        super().__init__(sa.create_engine(url))
        self.backend.database_name = url.database
