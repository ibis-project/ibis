"""The pandas backend is a departure from the typical ibis backend in that it
doesn't compile to anything, and the execution of the ibis expression
is under the purview of ibis itself rather than executing SQL against a server.

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
a different use case.

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

import collections
import datetime
import functools
import numbers
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import toolz

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.window as win
import ibis.pandas.aggcontext as agg_ctx
import ibis.util
from ibis.client import find_backends
from ibis.pandas.dispatch import (
    execute_first,
    execute_literal,
    execute_node,
    post_execute,
    pre_execute,
)

integer_types = np.integer, int
floating_types = (numbers.Real,)
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
date_types = (datetime.date,)
time_types = (datetime.time,)
timestamp_types = pd.Timestamp, datetime.datetime, np.datetime64
timedelta_types = pd.Timedelta, datetime.timedelta, np.timedelta64
temporal_types = date_types + time_types + timestamp_types + timedelta_types
scalar_types = fixed_width_types + temporal_types
simple_types = scalar_types + (str, type(None))


@functools.singledispatch
def is_computable_input(arg):
    """All inputs are not computable without a specific override."""
    return False


@is_computable_input.register(ibis.client.Client)
@is_computable_input.register(ir.Expr)
@is_computable_input.register(dt.DataType)
@is_computable_input.register(type(None))
@is_computable_input.register(win.Window)
@is_computable_input.register(tuple)
def is_computable_input_arg(arg):
    """Return whether `arg` is a valid computable argument."""
    return True


# Register is_computable_input for each scalar type (int, float, date, etc).
# We use consume here to avoid leaking the iteration variable into the module.
ibis.util.consume(
    is_computable_input.register(t)(is_computable_input_arg)
    for t in scalar_types
)


def get_node(obj):
    """Attempt to get the underlying :class:`Node` instance from `obj`."""
    try:
        return obj.op()
    except AttributeError:
        return obj


def dependencies(expr: ir.Expr):
    """Compute the dependencies of an expression.

    Parameters
    ----------
    expr
        An ibis expression

    Returns
    -------
    dict
        Mapping from hashable objects to ibis expression inputs.

    See Also
    --------
    is_computable_input
    dependents

    """
    stack = [expr]
    dependencies = collections.defaultdict(list)

    while stack:
        expr = stack.pop()
        node = get_node(expr)
        if isinstance(node, collections.abc.Hashable):
            if not isinstance(node, ops.Node):
                dependencies[node] = []
            if node not in dependencies:
                computable_inputs = [
                    arg for arg in node.inputs if is_computable_input(arg)
                ]
                stack.extend(computable_inputs)
                dependencies[node].extend(computable_inputs)
    return dict(dependencies)


def dependents(dependencies):
    """Get dependents from dependencies.

    Parameters
    ----------
    dependencies
        A mapping from hashable objects to ibis expression inputs.

    Returns
    -------
    dict
        A mapping from hashable objects to expressions that depend on the keys.

    See Also
    --------
    dependencies

    """
    dependents = collections.defaultdict(list)
    for node in dependencies.keys():
        dependents[node] = []

    for node, deps in dependencies.items():
        for dep in deps:
            dependents[get_node(dep)].append(node.to_expr())
    return dict(dependents)


def toposort(expr: ir.Expr):
    """Topologically sort the nodes that underly `expr`.

    Parameters
    ----------
    expr
        An ibis expression.

    Returns
    -------
    Tuple
        A tuple whose first element is the topologically sorted values required
        to compute `expr` and whose second element is the dependencies of
        `expr`.

    """
    # compute dependencies and dependents
    parents = dependencies(expr)
    children = dependents(parents)

    # count the number of dependencies each node has
    indegree = toolz.valmap(len, parents)

    # queue up the nodes with no dependencies
    queue = collections.deque(
        node for node, count in indegree.items() if not count
    )

    toposorted = []

    while queue:
        node = queue.popleft()

        # invariant: every element of the queue has indegree 0, i.e., no
        # dependencies
        assert not indegree[node]
        toposorted.append(node)

        # remove the node -> child edge for every child of node
        for child in map(get_node, children[node]):
            indegree[child] -= 1

            # if we removed the last edge, enqueue the child
            if not indegree[child]:
                queue.append(child)

    return toposorted, parents


def execute(
    expr: ir.Expr,
    scope: Optional[Mapping] = None,
    aggcontext: Optional[agg_ctx.AggregationContext] = None,
    clients: Sequence[ibis.client.Client] = (),
    params: Optional[Mapping] = None,
    **kwargs: Any
):
    """Execute an ibis expression against the pandas backend.

    Parameters
    ----------
    expr
    scope
    aggcontext
    clients
    params

    """
    toposorted, dependencies = toposort(expr)
    params = toolz.keymap(get_node, params if params is not None else {})

    # Add to scope the objects that have no dependencies and are not ibis
    # nodes. We have to filter out nodes for cases--such as zero argument
    # UDFs--that do not have any dependencies yet still need to be evaluated.
    full_scope = toolz.merge(
        scope if scope is not None else {},
        {
            key: key
            for key, parents in dependencies.items()
            if not parents and not isinstance(key, ops.Node)
        },
        params,
    )

    # Call pre_execute, to allow clients to intercept the expression before
    # computing anything *and* before associating leaf nodes with data. This
    # allows clients to provide their own data for each leaf.
    if not clients:
        clients = list(find_backends(expr))

    if aggcontext is None:
        aggcontext = agg_ctx.Summarize()

    # give backends a chance to inject scope if needed
    execute_first_scope = execute_first(
        expr.op(), *clients, scope=full_scope, aggcontext=aggcontext, **kwargs
    )
    full_scope = toolz.merge(full_scope, execute_first_scope)

    nodes = [node for node in toposorted if node not in full_scope]

    # compute the nodes that are not currently in scope
    for node in nodes:
        # allow clients to pre compute nodes as they like
        pre_executed_scope = pre_execute(
            node, *clients, scope=full_scope, aggcontext=aggcontext, **kwargs
        )
        # merge the existing scope with whatever was returned from pre_execute
        execute_scope = toolz.merge(full_scope, pre_executed_scope)

        # if after pre_execute our node is in scope, then there's nothing to do
        # in this iteration
        if node in execute_scope:
            full_scope = execute_scope
        else:
            # If we're evaluating a literal then we can be a bit quicker about
            # evaluating the dispatch graph
            if isinstance(node, ops.Literal):
                executor = execute_literal
            else:
                executor = execute_node

            # Gather the inputs we've already computed that the current node
            # depends on
            execute_args = [
                full_scope[get_node(arg)] for arg in dependencies[node]
            ]

            # execute the node with its inputs
            execute_node_result = executor(
                node,
                *execute_args,
                aggcontext=aggcontext,
                scope=execute_scope,
                clients=clients,
                **kwargs,
            )

            # last change to perform any additional computation on the result
            # before it gets added to scope for the next node
            full_scope[node] = post_execute(
                node,
                execute_node_result,
                clients=clients,
                aggcontext=aggcontext,
                scope=full_scope,
            )

    # the last node in the toposorted graph is the root and maps to the desired
    # result in scope
    last_node = toposorted[-1]
    result = full_scope[last_node]
    return result
