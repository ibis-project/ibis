from __future__ import annotations

import abc
import re
import warnings
from typing import Any, Callable

import ibis
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.common.exceptions import TranslationError

__all__ = ('BaseBackend', 'Database')


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


class BaseBackend(abc.ABC):
    """
    Base backend class.

    All Ibis backends are expected to subclass this `Backend` class,
    and implement all the required methods.
    """

    database_class = Database
    table_class: type[ops.DatabaseTable] = ops.DatabaseTable

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the backend, for example 'sqlite'.
        """

    @abc.abstractmethod
    def connect(connection_string, **options):
        """
        Connect to the underlying database and return a client object.
        """

    def database(self, name: str = None) -> Database:
        """
        Return a Database object for the `name` database.

        Parameters
        ----------
        name : str
            Name of the database to return the object for.

        Returns
        -------
        Database
            A database object for the specified database.
        """
        warnings.warn(
            'The `database` method and the `Database` object are '
            'deprecated and will be removed in a future version of Ibis. '
            'Use the equivalent methods in the backend instead.',
            FutureWarning,
        )
        return self.database_class(
            name=name or self.current_database, client=self
        )

    @property
    @abc.abstractmethod
    def current_database(self) -> str | None:
        """
        Name of the current database.

        Backends that don't support different databases will return None.

        Returns
        -------
        str
            Name of the current database.
        """

    @abc.abstractmethod
    def list_databases(self, like: str = None) -> list[str]:
        """
        List existing databases in the current connection.

        Parameters
        ----------
        like : str
            A pattern in Python's regex format to filter returned database
            names.

        Returns
        -------
        list of str
            The database names that exist in the current connection, that match
            the `like` pattern if provided.
        """

    def exists_database(self, name: str) -> bool:
        """
        Return whether a database name exists in the current connection.

        Deprecated in Ibis 2.0. Use `name in client.list_databases()` instead.
        """
        warnings.warn(
            '`client.exists_database(name)` is deprecated, and will be '
            'removed in a future version of Ibis. Use '
            '`name in client.list_databases()` instead.',
            FutureWarning,
        )
        return name in self.list_databases()

    @staticmethod
    def _filter_with_like(values: list[str], like: str = None) -> list[str]:
        """
        Filter names with a `like` pattern (regex).

        The methods `list_databases` and `list_tables` accept a `like`
        argument, which filters the returned tables with tables that match the
        provided pattern.

        We provide this method in the base backend, so backends can use it
        instead of reinventing the wheel.
        """
        if like is None:
            return values

        pattern = re.compile(like)
        return sorted(filter(lambda t: pattern.findall(t), values))

    @abc.abstractmethod
    def list_tables(self, like: str = None, database: str = None) -> list[str]:
        """
        Return the list of table names in the current database.

        For some backends, the tables may be files in a directory,
        or other equivalent entities in a SQL database.

        Parameters
        ----------
        like : str, optional
            A pattern in Python's regex format.
        database : str, optional
            The database to list tables of, if not the current one.

        Returns
        -------
        list of str
            The list of the table names that match the pattern `like`.
        """

    def exists_table(self, name: str, database: str = None) -> bool:
        """
        Return whether a table name exists in the database.

        Deprecated in Ibis 2.0. Use `name in client.list_tables()` instead.
        """
        warnings.warn(
            '`client.exists_table(name)` is deprecated, and will be '
            'removed in a future version of Ibis. Use '
            '`name in client.list_tables()` instead.',
            FutureWarning,
        )
        return len(self.list_tables(like=name, database=database)) > 0

    # @abc.abstractmethod
    def table(self, name: str, database: str = None) -> ir.TableExpr:
        """ """
        warnings.warn(
            '`database` argument of `.table()` is deprecated and '
            'will be removed in a future version of Ibis. Change '
            'the current database before calling `.table()` instead',
            FutureWarning,
        )

    def get_schema(self, table_name: str, database: str = None) -> sch.Schema:
        """
        Return the schema of `table_name`.

        Deprecated in Ibis 2.0. Use `.table(name).schema()` instead.
        """
        warnings.warn(
            '`.get_schema(name)` is deprecated, and will be '
            'removed in a future version of Ibis. Use '
            '`.table(name).schema()` instead',
            FutureWarning,
        )
        return self.table(name=table_name, database=database).schema()

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """
        Return the version of the backend engine.

        For database servers, that's the version of the PostgreSQL,
        MySQL,... server. For pandas, it would be the version of
        pandas, etc.
        """

    def register_options(self) -> None:
        """
        If the backend has custom options, register them here.
        They will be prefixed with the name of the backend.
        """

    def compile(self, expr: ir.Expr, params=None) -> Any:
        """
        Compile the expression.
        """
        return self.compiler.to_sql(expr, params=params)

    def execute(self, expr: ir.Expr) -> Any:  # XXX DataFrame for now?
        """ """

    def verify(self, expr: ir.Expr, params=None) -> bool:
        """
        Verify `expr` is an expression that can be compiled.
        """
        warnings.warn(
            '`verify` is deprecated, use `compile` and capture the '
            '`TranslationError` exception instead',
            FutureWarning,
        )
        try:
            self.compile(expr, params=params)
            return True
        except TranslationError:
            return False

    def add_operation(self, operation: ops.Node) -> Callable:
        """
        Decorator to add a translation function to the backend for a specific
        operation.

        Operations are defined in `ibis.expr.operations`, and a translation
        function receives the translator object and an expression as
        parameters, and returns a value depending on the backend. For example,
        in SQL backends, a NullLiteral operation could be translated simply
        with the string "NULL".

        Examples
        --------
        >>> @ibis.sqlite.add_operation(ibis.expr.operations.NullLiteral)
        ... def _null_literal(translator, expression):
        ...     return 'NULL'
        """
        if not hasattr(self, 'compiler'):
            raise RuntimeError(
                'Only SQL-based backends support `add_operation`'
            )

        def decorator(translation_function: Callable) -> None:
            self.compiler.translator_class.add_operation(
                operation, translation_function
            )

        return decorator

    def create_database(self, name: str, force: bool = False) -> None:
        """
        Create a new database.

        Not all backends implement this method.

        Parameters
        ----------
        name : str
            Name for the new database.
        force : bool, default False
            If `True`, an exception is raised if the database already exists.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "create_database"'
        )

    def create_table(
        self,
        name: str,
        obj=None,
        schema: ibis.Schema = None,
        database: str = None,
    ) -> None:
        """
        Create a new table.

        Not all backends implement this method.

        Parameters
        ----------
        name : str
            Name for the new table.
        obj : TableExpr or pandas.DataFrame, optional
            An Ibis or pandas table that will be used to extract the schema and
            the data of the new table. If not provided, `schema` must be.
        schema : ibis.Schema, optional
            The schema for the new table. Only one of `schema` or `obj` can be
            provided.
        database : str, optional
            Name of the database where the table will be created, if not the
            default.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "create_table"'
        )

    def drop_table(
        self, name: str, database: str = None, force: bool = False
    ) -> None:
        """
        Drop a table.

        Not all backends implement this method.

        Parameters
        ----------
        name : str
            Name of the table to drop.
        database : str, optional
            Name of the database where the table exists, if not the default.
        force : bool, default False
            If `True`, an exception is raised if the table does not exist.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "drop_table"'
        )

    def create_view(self, name: str, expr, database: str = None) -> None:
        """
        Create a new view.

        Not all backends implement this method.

        Parameters
        ----------
        name : str
            Name for the new view.
        expr : TableExpr
            An Ibis table expression that will be used to extract the query
            of the view.
        database : str, optional
            Name of the database where the view will be created, if not the
            default.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "create_view"'
        )

    def drop_view(
        self, name: str, database: str = None, force: bool = False
    ) -> None:
        """
        Drop a view.

        Not all backends implement this method.

        Parameters
        ----------
        name : str
            Name of the view to drop.
        database : str, optional
            Name of the database where the view exists, if not the default.
        force : bool, default False
            If `True`, an exception is raised if the view does not exist.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "drop_view"'
        )
