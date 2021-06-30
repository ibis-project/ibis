import abc

from ibis.common.exceptions import TranslationError

from .client import Client, Database
from .connection import BaseConnection

__all__ = ('BaseConnection', 'BaseBackend', 'Client', 'Database')


class BaseBackend(abc.ABC):
    """
    Base backend class.

    All Ibis backends are expected to subclass this `Backend` class,
    and implement all the required methods.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the backend, for example 'sqlite'.
        """
        pass

    @property
    @abc.abstractmethod
    def kind(self):
        """
        Backend kind. One of:

        sqlalchemy
            Backends using a SQLAlchemy dialect.
        sql
            SQL based backends, not based on a SQLAlchemy dialect.
        pandas
            Backends using pandas to store data and perform computations.
        spark
            Spark based backends.
        """
        pass

    @abc.abstractmethod
    def connect(connection_string, **options):
        """
        Connect to the underlying database and return a client object.
        """
        pass

    def register_options(self):
        """
        If the backend has custom options, register them here.
        They will be prefixed with the name of the backend.
        """
        pass

    def compile(self, expr, params=None):
        """
        Compile the expression.
        """
        return self.client.compiler.to_sql(expr, params=params)

    def verify(self, expr, params=None):
        """
        Verify `expr` is an expression that can be compiled.
        """
        try:
            self.compile(expr, params=params)
            return True
        except TranslationError:
            return False

    def add_operation(self, operation):
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
        if not hasattr(self.client, 'compiler'):
            raise RuntimeError(
                'Only SQL-based backends support `add_operation`'
            )

        def decorator(translation_function):
            self.client.compiler.translator_class.add_operation(
                operation, translation_function
            )

        return decorator
