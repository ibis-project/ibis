from __future__ import annotations

import ibis.expr.types as ir


class Client:
    """Generic Ibis client."""

    @property
    def version(self):
        return self.backend.version

    def list_tables(self, like=None, database=None):
        return self.backend.list_tables(like, database)

    def database(self, name=None):
        return self.backend.database(name)

    @property
    def current_database(self):
        return self.backend.current_database

    def list_databases(self, like=None):
        return self.backend.list_databases(like)

    def exists_database(self, name):
        return self.backend.exists_database(name)

    def exists_table(self, name, database=None):
        return self.backend.exists_table(name, database)

    # Methods below haven't been standardized among backends yet, #3019

    def attach(self, name, path, create=False):
        return self.backend.attach(name, path, create)

    def execute(self, *args, **kwargs):
        return self.backend.execute(*args, **kwargs)

    def table(self, *args, **kwargs):
        return self.backend.table(*args, **kwargs)

    def compile(self, *args, **kwargs):
        return self.backend.compile(*args, **kwargs)

    def create_table(self, *args, **kwargs):
        return self.backend.create_table(*args, **kwargs)

    def drop_table(self, *args, **kwargs):
        return self.backend.drop_table(*args, **kwargs)

    def raw_sql(self, *args, **kwargs):
        return self.backend.raw_sql(*args, **kwargs)

    def load_data(self, *args, **kwargs):
        return self.backend.load_data(*args, **kwargs)

    def inspector(self, *args, **kwargs):
        return self.backend.inspector(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.backend.insert(*args, **kwargs)

    def list_schemas(self, *args, **kwargs):
        return self.backend.list_schemas(*args, **kwargs)

    def ast_schema(self, *args, **kwargs):
        return self.backend.ast_schema(*args, **kwargs)

    def drop_database(self, *args, **kwargs):
        return self.backend.drop_database(*args, **kwargs)

    def truncate_table(self, *args, **kwargs):
        return self.backend.truncate_table(*args, **kwargs)

    def get_schema(self, *args, **kwargs):
        return self.backend.get_schema(*args, **kwargs)

    def sql(self, *args, **kwargs):
        return self.backend.sql(*args, **kwargs)

    def create_database(self, *args, **kwargs):
        return self.backend.create_database(*args, **kwargs)

    def avro_file(self, *args, **kwargs):
        return self.backend.avro_file(*args, **kwargs)

    def parquet_file(self, *args, **kwargs):
        return self.backend.parquet_file(*args, **kwargs)

    def set_database(self, *args, **kwargs):
        return self.backend.set_database(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.backend.close(*args, **kwargs)

    def disable_codegen(self, *args, **kwargs):
        return self.backend.disable_codegen(*args, **kwargs)

    @property
    def meta(self):
        return self.backend.meta

    @property
    def con(self):
        return self.backend.con

    @property
    def compiler(self):
        return self.backend.compiler


class Database:
    """Generic Database class."""

    def __init__(self, name, client):
        """Initialize the new object."""
        self.name = name
        self.client = client

    def __repr__(self) -> str:
        """Return type name and the name of the database."""
        return f'{type(self).__name__}({self.name!r})'

    def __dir__(self) -> list[str]:
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
        return self.client.list_tables(like, database=self.name)
