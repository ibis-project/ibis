"""The Dask backend.

The dask backend is a very close port of the pandas backend, and thus has
the similar caveats.

The dask backend is a departure from the typical ibis backend in that it
doesn't compile to anything and the execution of the ibis expression
is under the purview of ibis itself rather than executing SQL on a server.

Design
------
The dask backend uses a technique called `multiple dispatch
<https://en.wikipedia.org/wiki/Multiple_dispatch>`_, implemented in a
third-party open source library called `multipledispatch
<https://github.com/mrocklin/multipledispatch>`_.

Multiple dispatch is a generalization of standard single-dispatch runtime
polymorphism to multiple arguments.

Compilation
-----------
The ibis backend uses the ibis dispatching machinery to "compile" a dask
TaskGraph you can run `.compute()` on to evaluate it.

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

1. ``compute_time_context``
---------------------------
First, at the beginning of the main execution loop, ``compute_time_context`` is
called. This function computes time contexts, and pass them to all children of
the current node. These time contexts could be used in later steps to get data.
This is essential for time series Table, and related operations that adjust
time context, such as window, asof_join, etc.

By default, this function simply pass the unchanged time context to all
children nodes.


2. ``pre_execute``
------------------
Second, ``pre_execute`` is called.
This function serves a similar purpose to ``data_preload``, the key difference
being that ``pre_execute`` is called *every time* there's a call to execute.

By default this function does nothing.

3. ``execute_node``
-------------------

Then, when an expression is ready to be evaluated we call
:func:`~ibis.dask.core.execute` on the expressions arguments and then
:func:`~ibis.dask.dispatch.execute_node` on the expression with its
now-materialized arguments.

4. ``post_execute``
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

Defining a ``post_execute`` rule for :class:`~ibis.expr.operations.Window`
allows you to encode such logic. One might want to implement this using
:class:`~ibis.expr.operations.ScalarParameter`, in which case the ``scope``
passed to ``post_execute`` would be the bound values passed in at the time the
``execute`` method was called.


Scope
-----
Scope is used across the execution phases, it iss a map that maps Ibis
operators to actual data. It is used to cache data for calculated ops. It is
an optimization to reused executed results.

With time context included, the key is op associated with each expression;
And scope value is another key-value map:
- value: pd.DataFrame or pd.Series that is the result of executing key op
- timecontext: of type TimeContext, the time context associated with the data
stored in value

See ibis.common.scope for details about the implementation.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import TimeContext, canonicalize_context
from ibis.backends.dask import aggcontext as agg_ctx
from ibis.backends.dask.dispatch import (
    execute_literal,
    execute_node,
    post_execute,
    pre_execute,
)
from ibis.backends.dask.trace import trace
from ibis.backends.pandas.core import (
    compute_time_context,
    get_node_arguments,
    is_computable_input,
    is_computable_input_arg,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

is_computable_input.register(dd.core.Scalar)(is_computable_input_arg)


# TODO(kszucs): should deduplicate with pandas code since it is an exact copy
# of pandas.execute_with_scope()
def execute_with_scope(
    node: ops.Node,
    scope: Scope,
    timecontext: TimeContext | None = None,
    aggcontext: agg_ctx.AggregationContext | None = None,
    clients=None,
    **kwargs,
):
    """Execute an expression `expr`, with data provided in `scope`.

    Parameters
    ----------
    node
        The operation node to execute.
    scope
        A Scope class, with dictionary mapping `ops.Node` subclass instances to
        concrete data such as a pandas DataFrame.
    timecontext
        A tuple of (begin, end) that is passed from parent Node to children
        see [timecontext.py](ibis/backends/pandas/execution/timecontext.py) for
        detailed usage for this time context.
    aggcontext
        Context used to compute an aggregation.
    clients
        Sequence of clients
    kwargs
        Keyword arguments
    """
    # Call pre_execute, to allow clients to intercept the expression before
    # computing anything *and* before associating leaf nodes with data. This
    # allows clients to provide their own data for each leaf.
    if clients is None:
        clients, _ = node.to_expr()._find_backends()

    if aggcontext is None:
        aggcontext = agg_ctx.Summarize()

    pre_executed_scope = pre_execute(
        node,
        *clients,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    new_scope = scope.merge_scope(pre_executed_scope)
    result = execute_until_in_scope(
        node,
        new_scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        # XXX: we *explicitly* pass in scope and not new_scope here so that
        # post_execute sees the scope of execute_with_scope, not the scope of
        # execute_until_in_scope
        post_execute_=functools.partial(
            post_execute,
            scope=scope,
            timecontext=timecontext,
            aggcontext=aggcontext,
            clients=clients,
            **kwargs,
        ),
        **kwargs,
    ).get_value(node, timecontext)
    return result


# TODO(kszucs): should deduplicate with pandas code since it is an exact copy
# of pandas.execute_until_in_scope()
@trace
def execute_until_in_scope(
    node,
    scope: Scope,
    timecontext: TimeContext | None = None,
    aggcontext=None,
    clients=None,
    post_execute_=None,
    **kwargs,
) -> Scope:
    """Execute until our op is in `scope`."""
    # these should never be None
    assert aggcontext is not None, "aggcontext is None"
    assert clients is not None, "clients is None"
    assert post_execute_ is not None, "post_execute_ is None"

    # base case: our op has been computed (or is a leaf data node), so
    # return the corresponding value
    if scope.get_value(node, timecontext) is not None:
        return scope
    if isinstance(node, ops.Literal):
        # special case literals to avoid the overhead of dispatching
        # execute_node
        return Scope(
            {
                node: execute_literal(
                    node,
                    node.value,
                    node.dtype,
                    aggcontext=aggcontext,
                    **kwargs,
                )
            },
            timecontext,
        )

    # figure out what arguments we're able to compute on based on the
    # expressions inputs. things like expressions, None, and scalar types are
    # computable whereas ``list``s are not
    computable_args = [
        arg for arg in get_node_arguments(node) if is_computable_input(arg)
    ]

    # pre_executed_states is a list of states with same the length of
    # computable_args, these states are passed to each arg
    if timecontext:
        arg_timecontexts = compute_time_context(
            node,
            num_args=len(computable_args),
            timecontext=timecontext,
            clients=clients,
            scope=scope,
        )
    else:
        arg_timecontexts = [None] * len(computable_args)

    pre_executed_scope = pre_execute(
        node,
        *clients,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )

    new_scope = scope.merge_scope(pre_executed_scope)

    # Short circuit: if pre_execute puts op in scope, then we don't need to
    # execute its computable_args
    if new_scope.get_value(node, timecontext) is not None:
        return new_scope

    # recursively compute each node's arguments until we've changed type.
    # compute_time_context should return with a list with the same length
    # as computable_args, the two lists will be zipping together for
    # further execution
    if len(arg_timecontexts) != len(computable_args):
        raise com.IbisError(
            "arg_timecontexts differ with computable_arg in length "
            f"for type:\n{type(node).__name__}."
        )

    scopes = [
        execute_until_in_scope(
            arg,
            new_scope,
            timecontext=timecontext,
            aggcontext=aggcontext,
            post_execute_=post_execute_,
            clients=clients,
            **kwargs,
        )
        if isinstance(arg, ops.Node)
        else Scope({arg: arg}, timecontext)
        for (arg, timecontext) in zip(computable_args, arg_timecontexts)
    ]

    # if we're unable to find data then raise an exception
    if not scopes and computable_args:
        raise com.UnboundExpressionError(f"Unable to find data for node:\n{node!r}")

    # there should be exactly one dictionary per computable argument
    assert len(computable_args) == len(scopes)

    new_scope = new_scope.merge_scopes(scopes)
    # pass our computed arguments to this node's execute_node implementation
    data = [
        new_scope.get_value(arg, timecontext) if isinstance(arg, ops.Node) else arg
        for (arg, timecontext) in zip(computable_args, arg_timecontexts)
    ]
    result = execute_node(
        node,
        *data,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    computed = post_execute_(node, result, timecontext=timecontext)
    return Scope({node: computed}, timecontext)


execute = Dispatcher("execute")


@execute.register(ops.Node)
@trace
def main_execute(
    node: ops.Node,
    params: Mapping[ops.Node, Any] | None = None,
    scope: Scope | None = None,
    timecontext: TimeContext | None = None,
    aggcontext: agg_ctx.AggregationContext | None = None,
    cache: Mapping[ops.Node, Any] | None = None,
    **kwargs: Any,
):
    """Execute an expression against data that are bound to it.

    If no data are bound, raise a `ValueError`.

    Parameters
    ----------
    node : ibis.expr.operations.Node
        The operation node to execute
    params : Mapping[ibis.expr.operations.Node, object]
        The data that an unbound parameter in `expr` maps to
    scope : Mapping[ibis.expr.operations.Node, object]
        Additional scope, mapping ibis operations to data
    timecontext : Optional[TimeContext]
        timecontext needed for execution
    aggcontext : Optional[ibis.backends.pandas.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    cache
        Cache used to store computations.
    kwargs
        Additional arguments that can potentially be used by individual node
        execution

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """
    import ibis.expr.types as ir

    if scope is None:
        scope = Scope()

    if timecontext is not None:
        # convert timecontext to datetime type, if time strings are provided
        timecontext = canonicalize_context(timecontext)

    if params is None:
        params = {}

    if cache is None:
        cache = {}

    # TODO: make expressions hashable so that we can get rid of these .op()
    # calls everywhere
    params = {k.op() if isinstance(k, ir.Expr) else k: v for k, v in params.items()}
    scope = scope.merge_scope(Scope(params, timecontext))
    return execute_with_scope(
        node,
        scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        cache=cache,
        **kwargs,
    )


def execute_and_reset(
    node,
    params=None,
    scope=None,
    timecontext: TimeContext | None = None,
    aggcontext=None,
    **kwargs,
):
    """Execute an expression against data that are bound to it.

    If no data are bound, raise an Exception.

    The difference between this function and `ibis.dask.core.execute` is that
    this function resets the index of the result, if the result has an index.

    Parameters
    ----------
    node : ibis.expr.operations.Node
        The operation node to execute
    params : Mapping[ibis.expr.operations.Node, object]
        The data that an unbound parameter in `expr` maps to
    scope : Mapping[ibis.expr.operations.Node, object]
        Additional scope, mapping ibis operations to data
    timecontext : Optional[TimeContext]
        timecontext needed for execution
    aggcontext : Optional[ibis.dask.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    kwargs : Dict[str, object]
        Additional arguments that can potentially be used by individual node
        execution

    Returns
    -------
    result : Union[
        dask.dataframe.Series,
        dask.dataframe.DataFrame,
        ibis.dask.core.simple_types
    ]

    Raises
    ------
    ValueError
        * If no data are bound to the input expression.
    """
    result = execute(
        node,
        params=params,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    # Note - if `result` has npartitions > 1 `reset_index` will not create
    # a monotonically increasing index.
    if isinstance(result, dd.DataFrame):
        df = result.reset_index()
        return df[list(node.schema.names)]
    elif isinstance(result, dd.Series):
        return result.reset_index(drop=True)
    return result
