import ibis.backends.base
from ibis.backends.base_sqlalchemy.alchemy import AlchemyQueryBuilder

from .client import MySQLClient
from .compiler import MySQLDialect


class Backend(ibis.backends.base.BaseBackend):
    name = 'mysql'
    builder = AlchemyQueryBuilder
    dialect = MySQLDialect

    def connect(self):
        return MySQLClient(url=self.connection_string)
