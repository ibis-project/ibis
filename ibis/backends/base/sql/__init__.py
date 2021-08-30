from ibis.backends.base import BaseBackend

from .client import SQLClient

__all__ = ('SQLClient', 'BaseSQLBackend')


class BaseSQLBackend(BaseBackend):
    """
    Base backend class for backends that compile to SQL.
    """
