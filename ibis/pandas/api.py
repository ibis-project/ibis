from __future__ import absolute_import

import ibis
import toolz

from ibis.pandas.client import PandasClient
from ibis.pandas.decimal import execute_node  # noqa: F401
from ibis.pandas.execution import execute  # noqa: F401


__all__ = 'connect', 'execute', 'dialect'


def connect(dictionary):
    """Construct a pandas client from a dictionary of DataFrames.

    Parameters
    ----------
    dictionary : dict

    Returns
    -------
    PandasClient
    """
    return PandasClient(dictionary)


def from_dataframe(df, name='df', client=None):
    """
    convenience function to construct an ibis table
    from a DataFrame

    EXPERIMENTAL API

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
        return connect({name: df}).table(name)
    client.dictionary[name] = df
    return client.table(name)


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


class PandasExprTranslator(object):
    # get the dispatched functions from the execute_node dispatcher and compute
    # and flatten the type tree of the first argument which is always the Node
    # subclass
    _registry = frozenset(toolz.concat(
        _flatten_subclass_tree(types[0]) for types in execute_node.funcs
    ))
    _rewrites = {}


class PandasDialect(ibis.sql.compiler.Dialect):

    translator = PandasExprTranslator


PandasClient.dialect = dialect = PandasDialect
