import toolz

import ibis.config
from ibis.backends.base import BaseBackend

from .client import PandasClient, PandasDatabase, PandasTable
from .execution import execute, execute_node
from .udf import udf  # noqa F401


def _flatten_subclass_tree(cls):
    """Return the set of all child classes of `cls`.

    Parameters
    ----------
    cls : Type

    Returns
    -------
    frozenset[Type]
    """
    subclasses = frozenset(cls.__subclasses__())
    children = frozenset(toolz.concat(map(_flatten_subclass_tree, subclasses)))
    return frozenset({cls}) | subclasses | children


class PandasExprTranslator:
    # get the dispatched functions from the execute_node dispatcher and compute
    # and flatten the type tree of the first argument which is always the Node
    # subclass
    _registry = frozenset(
        toolz.concat(
            _flatten_subclass_tree(types[0]) for types in execute_node.funcs
        )
    )
    _rewrites = {}


class Backend(BaseBackend):
    name = 'pandas'
    kind = 'pandas'
    builder = None
    database_class = PandasDatabase
    table_class = PandasTable
    translator = PandasExprTranslator

    def connect(self, dictionary):
        """Construct a pandas client from a dictionary of DataFrames.

        Parameters
        ----------
        dictionary : dict

        Returns
        -------
        PandasClient
        """
        return PandasClient(backend=self, dictionary=dictionary)

    def execute(self, *args, **kwargs):
        return execute(*args, **kwargs)

    def from_dataframe(self, df, name='df', client=None):
        """
        convenience function to construct an ibis table
        from a DataFrame

        Parameters
        ----------
        df : DataFrame
        name : str, default 'df'
        client : Client, default new PandasClient
            client dictionary will be mutated with the
            name of the DataFrame

        Returns
        -------
        Table
        """

        if client is None:
            return self.connect({name: df}).table(name)
        client.dictionary[name] = df
        return client.table(name)

    def register_options(self):
        ibis.config.register_option(
            'enable_trace',
            False,
            'Whether enable tracing for pandas execution. '
            'See ibis.pandas.trace for details.',
            validator=ibis.config.is_bool,
        )
