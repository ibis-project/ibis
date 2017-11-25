from __future__ import absolute_import

import ibis
import toolz

from ibis.pandas.client import PandasClient
from ibis.pandas.decimal import execute_node  # noqa: F401
from ibis.pandas.execution import execute  # noqa: F401


__api__ = 'connect', 'execute'


def connect(dictionary):
    return PandasClient(dictionary)


def _flatten_subclass_tree(cls):
    subclasses = cls.__subclasses__()
    if not subclasses:
        return {cls}
    return frozenset(toolz.concat(
        map(_flatten_subclass_tree, subclasses)
    )) | frozenset(subclasses)


class PandasExprTranslator(object):
    _registry = frozenset(toolz.concat(
        _flatten_subclass_tree(types[0]) for types in execute_node.funcs
    ))
    _rewrites = {}


class PandasDialect(ibis.client.Dialect):

    translator = PandasExprTranslator


PandasClient.dialect = PandasDialect
