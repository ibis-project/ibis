from ibis.backends.base import BaseBackend

from .client import SQLClient

__all__ = ('SQLClient', 'BaseSQLBackend')


class BaseSQLBackend(BaseBackend):
    """
    Base backend class for backends that compile to SQL.
    """

    def list_tables(self, like=None, database=None):
        """
        By default we call `SHOW TABLES` against the database.

        Backends with other ways can overwrite this method.
        """
        return self._filter_with_like(
            [row[0] for row in self.client.raw_sql('SHOW TABLES').fetchall()]
        )
