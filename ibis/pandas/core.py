"""The pandas backend is a departure from the typical ibis backend in that it
doesn't compile to anything, and the execution of the ibis expression coming
dervied from a PandasTable is under the purview of ibis itself rather than
sending something to a server and having it return the results.

Design
------

Ibis uses a technique called `multiple dispatch
<https://en.wikipedia.org/wiki/Multiple_dispatch>`_, implemented in a
third-party open source library called `multipledispatch
<https://github.com/mrocklin/multipledispatch>`_.

Multiple dispatch is a generalization of standard single-dispatch runtime
polymorphism to multiple arguments.

Compilation
-----------
This is a no-op because we directly execute ibis expressions.

Execution
---------

Execution is divided into different dispatched functions, each arising from
different a use case.

A top level dispatched function named ``execute`` with two signatures exists
to provide a single API for executing an ibis expression.

The general flow of execution is:

::
       If the current operation is in scope:
           return it
       Else:
           execute the arguments of the current node

       execute the current node with its executed arguments

Specifically, execute is comprised of 4 steps that happen at different times
during the loop.

1. ``data_preload``
-------------------
First, data_preload is called. data_preload provides a way for an expression to
intercept the data and inject scope. By default it does nothing.

2. ``pre_execute``
------------------

Second, at the beginning of the main execution loop, ``pre_execute`` is called.
This function serves a similar purpose to ``data_preload``, the key difference
being that ``pre_execute`` is called *every time* there's a call to execute.

By default this function does nothing.

3. ``execute_first``
--------------------

Third is ``execute_first``. This function gives an expression the opportunity
to define the entire flow of execution of an expression starting from the top
of the expression.

This functionality was essential for implementing window functions in the
pandas backend

By default this function does nothing.

4. ``execute_node``
-------------------

Finally, when an expression is ready to be evaluated we call
:func:`~ibis.pandas.core.execute` on the expressions arguments and then
:func:`~ibis.pandas.dispatch.execute_node` on the expression with its
now-materialized arguments.
"""

from __future__ import absolute_import

import collections
import numbers
import datetime

import six

import numpy as np

import toolz

import ibis.expr.types as ir
import ibis.expr.lineage as lin
import ibis.expr.datatypes as dt

from ibis.compat import functools
from ibis.client import find_backends

import ibis.pandas.aggcontext as agg_ctx
from ibis.pandas.dispatch import (
    execute, execute_node, execute_first, data_preload, pre_execute
)


integer_types = six.integer_types + (np.integer,)
floating_types = numbers.Real,
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
temporal_types = (
    datetime.datetime, datetime.date, datetime.timedelta, datetime.time,
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
    def finder(expr):
        op = expr.op()
        if hasattr(op, 'source'):
            data = (op, op.source.dictionary.get(op.name, None))
        else:
            data = None
        return lin.proceed, data

    return collections.OrderedDict(lin.traverse(finder, expr))


_VALID_INPUT_TYPES = (ir.Expr, dt.DataType, type(None)) + scalar_types


@execute.register(ir.Expr, dict)
def execute_with_scope(expr, scope, context=None, **kwargs):
    """Execute an expression `expr`, with data provided in `scope`.

    Parameters
    ----------
    expr : ir.Expr
        The expression to execute.
    scope : dict
        A dictionary mapping :class:`~ibis.expr.types.Node` subclass instances
        to concrete data such as a pandas DataFrame.

    Returns
    -------
    result : scalar, pd.Series, pd.DataFrame
    """
    op = expr.op()

    # Call pre_execute, to allow clients to intercept the expression before
    # computing anything *and* before associating leaf nodes with data. This
    # allows clients to provide their own scope.
    scope = toolz.merge(
        scope,
        *map(
            functools.partial(pre_execute, op, scope=scope, **kwargs),
            find_backends(expr)
        )
    )

    # base case: our op has been computed (or is a leaf data node), so
    # return the corresponding value
    if op in scope:
        return scope[op]

    if context is None:
        context = agg_ctx.Summarize()

    try:
        computed_args = [scope[t] for t in op.root_tables()]
    except KeyError:
        pass
    else:
        try:
            # special case: we have a definition of execute_first that matches
            # our current operation and data leaves
            return execute_first(
                op, *computed_args, scope=scope, context=context, **kwargs
            )
        except NotImplementedError:
            pass

    args = op.args

    # recursively compute the op's arguments
    computed_args = [
        execute(arg, scope, context=context, **kwargs)
        if hasattr(arg, 'op') else arg
        for arg in args if isinstance(arg, _VALID_INPUT_TYPES)
    ]

    # Compute our op, with its computed arguments
    return execute_node(
        op, *computed_args,
        scope=scope,
        context=context,
        **kwargs
    )


@execute.register(ir.Expr)
def execute_without_scope(
        expr, params=None, scope=None, context=None, **kwargs):
    """Execute an expression against data that are bound to it. If no data
    are bound, raise an Exception.

    Parameters
    ----------
    expr : ir.Expr
        The expression to execute
    params : Dict[Expr, object]

    Returns
    -------
    result : scalar, pd.Series, pd.DataFrame

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """

    data_scope = find_data(expr)

    factory = type(data_scope)

    if scope is None:
        scope = factory()

    if params is None:
        params = factory()

    params = {k.op() if hasattr(k, 'op') else k: v for k, v in params.items()}

    new_scope = toolz.merge(scope, data_scope, params, factory=factory)

    # data_preload
    new_scope.update(
        (node, data_preload(node, data, scope=new_scope))
        for node, data in new_scope.items()
    )

    # By default, our aggregate functions are N -> 1
    return execute(
        expr,
        new_scope,
        context=context if context is not None else agg_ctx.Summarize(),
        **kwargs
    )
