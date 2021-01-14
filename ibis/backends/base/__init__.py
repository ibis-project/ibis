import abc

import ibis.config
from ibis.common.exceptions import TranslationError


class BaseBackend(abc.ABC):
    """
    Base backend class.

    All Ibis backends are expected to subclass this `Backend` class,
    and implement all the required methods.
    """

    def __init__(self, connection_string):
        self.connection_string = connection_string

        with ibis.config.config_prefix(self.name):
            self.register_options()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the backend, for example 'sqlite'.
        """
        pass

    @property
    @abc.abstractmethod
    def builder(self):
        pass

    @property
    @abc.abstractmethod
    def dialect(self):
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
        context = self.dialect.make_context(params=params)
        builder = self.builder(expr, context=context)
        query_ast = builder.get_result()
        # TODO bigquery doesn't use [0]
        compiled = query_ast[0].compile()
        return compiled

    def verify(self, expr, params=None):
        """
        Verify `expr` is an expression that can be compiled.
        """
        try:
            self.compile(expr, params=params)
            return True
        except TranslationError:
            return False
