import ibis.backends.base
from ibis.backends.base_sqlalchemy.alchemy import AlchemyQueryBuilder

from .client import PostgreSQLClient
from .compiler import PostgreSQLDialect


class Backend(ibis.backends.base.BaseBackend):
    name = 'postgres'
    builder = AlchemyQueryBuilder
    dialect = PostgreSQLDialect

    def connect(self):
        return PostgreSQLClient(url=self.connection_string)
