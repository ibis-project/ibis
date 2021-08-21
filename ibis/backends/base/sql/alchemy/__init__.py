from ibis.backends.base.sql import BaseSQLBackend

from .client import AlchemyClient
from .database import AlchemyDatabase, AlchemyDatabaseSchema, AlchemyTable
from .datatypes import schema_from_table, table_from_schema, to_sqla_type
from .query_builder import AlchemyCompiler
from .registry import (
    fixed_arity,
    get_sqla_table,
    infix_op,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
    varargs,
    variance_reduction,
)
from .translator import AlchemyContext, AlchemyExprTranslator

__all__ = (
    'BaseAlchemyBackend',
    'AlchemyExprTranslator',
    'AlchemyContext',
    'AlchemyCompiler',
    'AlchemyClient',
    'AlchemyTable',
    'AlchemyDatabaseSchema',
    'AlchemyDatabase',
    'AlchemyContext',
    'sqlalchemy_operation_registry',
    'sqlalchemy_window_functions_registry',
    'reduction',
    'variance_reduction',
    'fixed_arity',
    'unary',
    'infix_op',
    'get_sqla_table',
    'to_sqla_type',
    'schema_from_table',
    'table_from_schema',
    'varargs',
)


class BaseAlchemyBackend(BaseSQLBackend):
    """
    Base backend class for backends that compile to SQL with SQLAlchemy.
    """

    database_class = AlchemyDatabase
    table_class = AlchemyTable

    @property
    def version(self):
        return '.'.join(map(str, self.client.con.dialect.server_version_info))

    def list_databases(self, like=None):
        """List databases in the current server."""
        databases = self.client.inspector.get_schema_names()
        return self._filter_with_like(databases, like)
