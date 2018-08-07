"""The pandas backend is a departure from the typical ibis backend in that it
doesn't compile to anything, and the execution of the ibis expression
is under the purview of ibis itself rather than executing SQL on a server.

Design
------
The pandas backend uses a technique called `multiple dispatch
<https://en.wikipedia.org/wiki/Multiple_dispatch>`_, implemented in a
third-party open source library called `multipledispatch
<https://github.com/mrocklin/multipledispatch>`_.

Multiple dispatch is a generalization of standard single-dispatch runtime
polymorphism to multiple arguments.

Compilation
-----------
This is a no-op because we execute ibis expressions directly.

Execution
---------
Execution is divided into different dispatched functions, each arising from
different a use case.

A top level function `execute` exists to provide the API for executing an ibis
expression against in-memory data.

The general flow of execution is:

::
       If the current operation is in scope:
           return it
       Else:
           execute the arguments of the current node

       execute the current node with its executed arguments

Specifically, execute is comprised of a series of steps that happen at
different times during the loop.

1. ``pre_execute``
------------------
First, at the beginning of the main execution loop, ``pre_execute`` is called.
This function serves a similar purpose to ``data_preload``, the key difference
being that ``pre_execute`` is called *every time* there's a call to execute.

By default this function does nothing.

2. ``execute_node``
-------------------

Second, when an expression is ready to be evaluated we call
:func:`~ibis.pandas.core.execute` on the expressions arguments and then
:func:`~ibis.pandas.dispatch.execute_node` on the expression with its
now-materialized arguments.

3. ``post_execute``
-------------------
The final step--``post_execute``--is called immediately after the previous call
to ``execute_node`` and takes the instance of the
:class:`~ibis.expr.operations.Node` just computed and the result of the
computation.

The purpose of this function is to allow additional computation to happen in
the context of the current level of the execution loop. You might be wondering
That may sound vague, so let's look at an example.

Let's say you want to take a three day rolling average, and you want to include
3 days of data prior to the first date of the input. You don't want to see that
data in the result for a few reasons, one of which is that it would break the
contract of window functions: given N rows of input there are N rows of output.

Defining a ``post_execute`` rule for :class:`~ibis.expr.operations.WindowOp`
allows you to encode such logic. One might want to implement this using
:class:`~ibis.expr.operations.ScalarParameter`, in which case the ``scope``
passed to ``post_execute`` would be the bound values passed in at the time the
``execute`` method was called.
"""

from __future__ import absolute_import

import numbers
import datetime

import six

import numpy as np

import pandas as pd

import toolz

from multipledispatch import Dispatcher

import ibis
import ibis.common as com

import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.expr.datatypes as dt
import ibis.expr.window as win

from ibis.compat import functools
from ibis.client import find_backends

import ibis.pandas.aggcontext as agg_ctx
from ibis.pandas.dispatch import (
     execute_node, pre_execute, post_execute, execute_literal
)


integer_types = six.integer_types + (np.integer,)
floating_types = numbers.Real,
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
date_types = datetime.date,
time_types = datetime.time,
timestamp_types = pd.Timestamp, datetime.datetime, np.datetime64
timedelta_types = pd.Timedelta, datetime.timedelta, np.timedelta64
temporal_types = date_types + time_types + timestamp_types + timedelta_types
scalar_types = fixed_width_types + temporal_types
simple_types = scalar_types + six.string_types

_VALID_INPUT_TYPES = (
    ibis.client.Client, ir.Expr, dt.DataType, type(None), win.Window, tuple
) + scalar_types


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
    result : scalar, pd.Series, pd.DataFrame
    """
    op = expr.op()

    # Call pre_execute, to allow clients to intercept the expression before
    # computing anything *and* before associating leaf nodes with data. This
    # allows clients to provide their own data for each leaf.
    if clients is None:
        clients = list(find_backends(expr))

    if aggcontext is None:
        aggcontext = agg_ctx.Summarize()

    pre_executed_scope = pre_execute(
        op, *clients, scope=scope, aggcontext=aggcontext, **kwargs)
    new_scope = toolz.merge(scope, pre_executed_scope)
    result = execute_until_in_scope(
        expr,
        new_scope,
        aggcontext=aggcontext,
        clients=clients,

        # XXX: we *explicitly* pass in scope and not new_scope here so that
        # post_execute sees the scope of execute_with_scope, not the scope of
        # execute_until_in_scope
        post_execute_=functools.partial(
            post_execute,
            scope=scope,
            aggcontext=aggcontext,
            clients=clients,
            **kwargs
        ),
        **kwargs
    )

    return result


def execute_until_in_scope(
    expr, scope, aggcontext=None, clients=None, post_execute_=None, **kwargs
):
    """Execute until our op is in `scope`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
    scope : Mapping
    aggcontext : Optional[AggregationContext]
    clients : List[ibis.client.Client]
    kwargs : Mapping
    """
    # these should never be None
    assert aggcontext is not None, 'aggcontext is None'
    assert clients is not None, 'clients is None'
    assert post_execute_ is not None, 'post_execute_ is None'

    # base case: our op has been computed (or is a leaf data node), so
    # return the corresponding value
    op = expr.op()
    if op in scope:
        return scope[op]

    new_scope = execute_bottom_up(
        expr, scope,
        aggcontext=aggcontext,
        post_execute_=post_execute_, clients=clients, **kwargs)
    new_scope = toolz.merge(
        new_scope, pre_execute(op, *clients, scope=scope, **kwargs)
    )
    return execute_until_in_scope(
        expr, new_scope,
        aggcontext=aggcontext, clients=clients, post_execute_=post_execute_,
        **kwargs
    )


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


def execute_bottom_up(
    expr, scope, aggcontext=None, post_execute_=None, clients=None, **kwargs
):
    """Execute `expr` bottom-up.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
    scope : Mapping[ibis.expr.operations.Node, object]
    aggcontext : Optional[ibis.pandas.aggcontext.AggregationContext]
    kwargs : Dict[str, object]

    Returns
    -------
    result : Mapping[
        ibis.expr.operations.Node,
        Union[pandas.Series, pandas.DataFrame, scalar_types]
    ]
        A mapping from node to the computed result of that Node
    """
    assert post_execute_ is not None, 'post_execute_ is None'
    op = expr.op()

    # if we're in scope then return the scope, this will then be passed back
    # into execute_bottom_up, which will then terminate
    if op in scope:
        return scope
    elif isinstance(op, ops.Literal):
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
            post_execute_=post_execute_,
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
    computed = post_execute_(op, result)
    return {op: computed}


execute = Dispatcher('execute')


@execute.register(ir.Expr)
def main_execute(expr, params=None, scope=None, aggcontext=None, **kwargs):
    """Execute an expression against data that are bound to it. If no data
    are bound, raise an Exception.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The expression to execute
    params : Mapping[ibis.expr.types.Expr, object]
        The data that an unbound parameter in `expr` maps to
    scope : Mapping[ibis.expr.operations.Node, object]
        Additional scope, mapping ibis operations to data
    aggcontext : Optional[ibis.pandas.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    kwargs : Dict[str, object]
        Additional arguments that can potentially be used by individual node
        execution

    Returns
    -------
    result : Union[
        pandas.Series, pandas.DataFrame, ibis.pandas.core.simple_types
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


def execute_and_reset(
    expr, params=None, scope=None, aggcontext=None, **kwargs
):
    """Execute an expression against data that are bound to it. If no data
    are bound, raise an Exception.

    Notes
    -----
    The difference between this function and :func:`~ibis.pandas.core.execute`
    is that this function resets the index of the result, if the result has
    an index.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The expression to execute
    params : Mapping[ibis.expr.types.Expr, object]
        The data that an unbound parameter in `expr` maps to
    scope : Mapping[ibis.expr.operations.Node, object]
        Additional scope, mapping ibis operations to data
    aggcontext : Optional[ibis.pandas.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    kwargs : Dict[str, object]
        Additional arguments that can potentially be used by individual node
        execution

    Returns
    -------
    result : Union[
        pandas.Series, pandas.DataFrame, ibis.pandas.core.simple_types
    ]

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """
    result = execute(
        expr, params=params, scope=scope, aggcontext=aggcontext, **kwargs)
    if isinstance(result, pd.DataFrame):
        schema = expr.schema()
        return result.reset_index()[schema.names]
    elif isinstance(result, pd.Series):
        return result.reset_index(drop=True)
    return result
