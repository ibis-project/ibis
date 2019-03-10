"""
Core execution functions for the arrow backend.
"""
import datetime

import numbers

import numpy as np

import six

import toolz

import pyarrow as pa

import ibis

from ibis.client import find_backends
from ibis.compat import functools

import ibis.common as com
import ibis.expr.operations as ops
import ibis.expr.types as ir

from multipledispatch import Dispatcher

from arrow.dispatch import execute_node, execute_literal

import arrow.aggcontext as agg_ctx

integer_types = six.integer_types + (np.integer,)
floating_types = numbers.Real,
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
date_types = datetime.date,
time_types = datetime.time,
timestamp_types = datetime.datetime, np.datetime64
timedelta_types = pa.time32, pa.time64, pa.timestamp, datetime.timedelta, np.timedelta64
temporal_types = date_types + time_types + timestamp_types # + timedelta_types
scalar_types = fixed_width_types + temporal_types
simple_types = scalar_types + six.string_types

_VALID_INPUT_TYPES = (ibis.client.Client, ir.Expr, type(None))


def execute_with_scope(expr, scope, aggcontext=None, clients=None, **kwargs):
    """Execute an expression `expr`, with data provided in `scope`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The expression to execute.
    scope : collections.Mapping
        A dictionary mapping :class:`~ibis.expr.operations.Node` subclass
        instances to concrete data such as a pandas DataFrame.
    aggcontext : Optional[ibis.pandas.aggcontext.AggregationContext]

    Returns
    -------
    result : pa.RecordBatch
    """
    if clients is None:
        clients = list(find_backends(expr))

    if aggcontext is None:
        aggcontext = agg_ctx.Summarize()

    new_scope = scope
    result = execute_until_in_scope(
        expr,
        new_scope,
        aggcontext=aggcontext,
        clients=clients,
        post_execute=None,
        **kwargs
    )
    return result


def execute_until_in_scope(expr, scope, aggcontext=None, clients=None, **kwargs):
    """Execute until our op is in `scope`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
    scope : Mapping
    """
    # base case: our op has been computed (or is a leaf data node), so
    # return the corresponding value
    op = expr.op()
    if op in scope:
        return scope[op]

    new_scope = execute_bottom_up(expr, scope, aggcontext=aggcontext)
    return execute_until_in_scope(expr, new_scope)


def is_computable_arg(op, arg):
    """Is `arg` a valid input to an ``execute_node`` rule?

    Parameters
    ----------
    arg : object
        Any Python object

    Returns
    -------
    result : bool
    """
    return (
        isinstance(op, (ops.ExpressionList, ops.ValueList, ops.WindowOp)) or
        isinstance(arg, _VALID_INPUT_TYPES)
    )


def execute_bottom_up(expr, scope, aggcontext=None, post_execute_=None, clients=None, **kwargs):
    """Execute `expr` bottom-up.

    Parameters
    ----------
    expr : ibis.expr.types.Expr

    Returns
    -------
    result : Mapping[
        ibis.expr.operations.Node,
        Union[pandas.Series, pandas.DataFrame, scalar_types]
    ]
        A mapping from node to the computed result of that Node
    """

    # assert post_execute_ is not None, 'post_execute_ is None'
    op = expr.op()

    # if we're in scope then return the scope, this will then be passed back
    # into execute_bottom_up, which will then terminate
    if op in scope:
        return scope
    if isinstance(op, ops.Literal):
        # special case literals to avoid the overhead of dispatching
        # execute_node
        return {
            op: execute_literal(
                op, op.value, expr.type(), aggcontext=aggcontext, **kwargs
            )
        }

    # figure out what arguments we're able to compute on based on the
    # expressions inputs. things like expressions, None, and scalar types are
    # computable whereas ``list``s are not
    args = op.inputs
    is_computable_argument = functools.partial(is_computable_arg, op)
    computable_args = list(filter(is_computable_argument, args))

    # recursively compute each node's arguments until we've changed type
    scopes = [
        execute_bottom_up(
            arg, scope,
            aggcontext=aggcontext,
            clients=clients,
            **kwargs)
        if hasattr(arg, 'op') else {arg: arg}
        for arg in computable_args
    ]

    # if we're unable to find data then raise an exception
    if not scopes:
        raise com.UnboundExpressionError(
            'Unable to find data for expression:\n{}'.format(repr(expr))
        )

    # there should be exactly one dictionary per computable argument
    assert len(computable_args) == len(scopes)

    new_scope = toolz.merge(scopes)

    # pass our computed arguments to this node's execute_node implementation
    data = [
        new_scope[arg.op()] if hasattr(arg, 'op') else arg
        for arg in computable_args
    ]
    result = execute_node(
        op, *data,
        scope=scope, aggcontext=aggcontext, clients=clients, **kwargs)
    # result = execute_node(op, *data, scope=scope)
    return {op: result}


execute = Dispatcher('execute')


@execute.register(ir.Expr)
def main_execute(expr, params=None, scope=None, aggcontext=None, **kwargs):
    """Execute an expression against data that are bound to it. If no data
    are bound, raise an Exception.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The expression to execute

    Returns
    -------
    result : Union[
        pa.RecordBatch, ibis.pandas.core.simple_types
    ]

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """

    if scope is None:
        scope = {}

    if params is None:
        params = {}

    # TODO: make expresions hashable so that we can get rid of these .op()
    # calls everywhere
    params = {k.op() if hasattr(k, 'op') else k: v for k, v in params.items()}

    new_scope = toolz.merge(scope, params)
    return execute_with_scope(expr, new_scope, aggcontext=aggcontext, **kwargs)
