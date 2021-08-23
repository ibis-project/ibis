from __future__ import annotations

from typing import List

import ibis.expr.types as ir


class Client:
    """Generic Ibis client."""

    @property
    def version(self):
        return self.backend.version

    def list_tables(self, like=None):
        return self.backend.list_tables(like)

    def database(self, name=None):
        return self.backend.database(name)

    @property
    def current_database(self):
        return self.backend.current_database

    def list_databases(self, like=None):
        return self.backend.list_databases(like)

    def exists_database(self, name):
        return self.backend.exists_database(name)


class Database:
    """Generic Database class."""

    def __init__(self, name, client):
        """Initialize the new object."""
        self.name = name
        self.client = client

    def __repr__(self) -> str:
        """Return type name and the name of the database."""
        return '{}({!r})'.format(type(self).__name__, self.name)

    def __dir__(self) -> List[str]:
        """Return a set of attributes and tables available for the database.

        Returns
        -------
        set
            A set of the attributes and tables available for the database.
        """
        attrs = dir(type(self))
        unqualified_tables = [self._unqualify(x) for x in self.tables]
        return sorted(frozenset(attrs + unqualified_tables))

    def __contains__(self, key: str) -> bool:
        """
        Check if the given table (key) is available for the current database.

        Parameters
        ----------
        key : string

        Returns
        -------
        bool
            True if the given key (table name) is available for the current
            database.
        """
        return key in self.tables

    @property
    def tables(self) -> list:
        """Return a list with all available tables.

        Returns
        -------
        list
        """
        return self.list_tables()

    def __getitem__(self, key: str) -> ir.TableExpr:
        """Return a TableExpr for the given table name (key).

        Parameters
        ----------
        key : string

        Returns
        -------
        TableExpr
        """
        return self.table(key)

    def __getattr__(self, key: str) -> ir.TableExpr:
        """Return a TableExpr for the given table name (key).

        Parameters
        ----------
        key : string

        Returns
        -------
        TableExpr
        """
        return self.table(key)

    def _qualify(self, value):
        return value

    def _unqualify(self, value):
        return value

    def drop(self, force: bool = False):
        """Drop the database.

        Parameters
        ----------
        force : boolean, default False
          If True, Drop any objects if they exist, and do not fail if the
          databaes does not exist.
        """
        self.client.drop_database(self.name, force=force)

    def table(self, name: str) -> ir.TableExpr:
        """Return a table expression referencing a table in this database.

        Parameters
        ----------
        name : string

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._qualify(name)
        return self.client.table(qualified_name, self.name)

    def list_tables(self, like=None):
        return self.client.list_tables(like)
