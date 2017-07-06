from __future__ import absolute_import

import collections
import numbers
import datetime

import six

import numpy as np

import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from ibis.pandas.dispatch import execute, execute_node


integer_types = six.integer_types + (np.integer,)
floating_types = numbers.Real,
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
temporal_types = (
    datetime.datetime, datetime.date, datetime.timedelta,
    np.datetime64, np.timedelta64,
)
scalar_types = fixed_width_types + temporal_types
simple_types = scalar_types + six.string_types


def find_data(expr):
    """Find data sources bound to `expr`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr

    Returns
    -------
    data : collections.OrderedDict
    """
    stack = [expr]
    seen = set()
    data = collections.OrderedDict()

    while stack:
        e = stack.pop()
        node = e.op()

        if node not in seen:
            seen.add(node)

            if hasattr(node, 'source'):
                data[e] = node.source.dictionary[node.name]
            elif isinstance(node, ir.Literal):
                data[e] = node.value

            stack.extend(arg for arg in node.args if isinstance(arg, ir.Expr))
    return data


_VALID_INPUT_TYPES = (ir.Expr, dt.DataType, type(None)) + scalar_types


@execute.register(ir.Expr, dict)
def execute_with_scope(expr, scope):
    if expr in scope:
        return scope[expr]

    op = expr.op()
    args = op.args

    computed_args = [
        execute(arg, scope) if hasattr(arg, 'op') else arg
        for arg in args if isinstance(arg, _VALID_INPUT_TYPES)
    ] or [scope.get(arg, arg) for arg in args]

    return execute_node(op, *computed_args, scope=scope)


@execute.register(ir.Expr)
def execute_without_scope(expr):
    scope = find_data(expr)
    if not scope:
        raise ValueError('No data sources found')
    return execute(expr, scope)
