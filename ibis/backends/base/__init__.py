import abc
import warnings
from typing import Any, Callable, List

import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.common.exceptions import TranslationError

from .client import Client, Database

__all__ = ('BaseBackend', 'Client', 'Database')


class BaseBackend(abc.ABC):
    """
    Base backend class.

    All Ibis backends are expected to subclass this `Backend` class,
    and implement all the required methods.
    """

    database_class = Database
    table_class = ops.DatabaseTable

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the backend, for example 'sqlite'.
        """

    @property
    @abc.abstractmethod
    def client_class(self):
        """Class of the client, like `PandasClient`."""

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
            name=name or self.current_database(), client=self.client
        )

    # @abc.abstractmethod
    def current_database(self) -> str:
        """
        """
        return self.client.current_database()

    # @abc.abstractmethod
    def list_tables(self, like: str = None) -> List[str]:
        """
        """

    # @abc.abstractmethod
    def table(self, name: str, database: str = None) -> ir.TableExpr:
        """
        """
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
        return self.client_class.compiler.to_sql(expr, params=params)

    def execute(self, expr: ir.Expr) -> Any:  # XXX DataFrame for now?
        """
        """

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
        if not hasattr(self.client_class, 'compiler'):
            raise RuntimeError(
                'Only SQL-based backends support `add_operation`'
            )

        def decorator(translation_function: Callable) -> None:
            self.client_class.compiler.translator_class.add_operation(
                operation, translation_function
            )

        return decorator
