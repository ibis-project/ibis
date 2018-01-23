from __future__ import absolute_import

import ibis
import toolz

from ibis.pandas.client import PandasClient
from ibis.pandas.decimal import execute_node  # noqa: F401
from ibis.pandas.execution import execute  # noqa: F401


__all__ = 'connect', 'execute'


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


class PandasDialect(ibis.client.Dialect):

    translator = PandasExprTranslator


PandasClient.dialect = PandasDialect
