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
First, at the beginning of the main execution loop, ``compute_time_context`` is
called. This function computes time contexts, and pass them to all children of
the current node. These time contexts could be used in later steps to get data.
This is essential for time series TableExpr, and related operations that adjust
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
:func:`~ibis.backends.pandas.core.execute` on the expressions arguments and then
:func:`~ibis.backends.pandas.dispatch.execute_node` on the expression with its
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

Defining a ``post_execute`` rule for :class:`~ibis.expr.operations.WindowOp`
allows you to encode such logic. One might want to implement this using
:class:`~ibis.expr.operations.ScalarParameter`, in which case the ``scope``
passed to ``post_execute`` would be the bound values passed in at the time the
``execute`` method was called.


Scope
-------------------
Scope is used across the execution phases, it iss a map that maps Ibis
operators to actual data. It is used to cache data for calculated ops. It is
an optimization to reused executed results.

With time context included, the key is op associated with each expression;
And scope value is another key-value map:
- value: pd.DataFrame or pd.Series that is the result of executing key op
- timecontext: of type TimeContext, the time context associated with the data
stored in value

See ibis.common.scope for details about the implementaion.
"""

from __future__ import absolute_import

import datetime
import functools
import numbers
from typing import Optional

import numpy as np
import pandas as pd
from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.window as win
import ibis.util
from ibis.client import find_backends
from ibis.expr.scope import Scope
from ibis.expr.timecontext import canonicalize_context
from ibis.expr.typing import TimeContext

from . import aggcontext as agg_ctx
from .dispatch import execute_literal, execute_node, post_execute, pre_execute
from .trace import trace

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


def execute_with_scope(
    expr,
    scope: Scope,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    """Execute an expression `expr`, with data provided in `scope`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The expression to execute.
    scope : Scope
        A Scope class, with dictionary mapping
        :class:`~ibis.expr.operations.Node` subclass instances to concrete
        data such as a pandas DataFrame.
    timecontext : Optional[TimeContext]
        A tuple of (begin, end) that is passed from parent Node to children
        see [timecontext.py](ibis/backends/pandas/execution/timecontext.py) for
        detailed usage for this time context.
    aggcontext : Optional[ibis.backends.pandas.aggcontext.AggregationContext]

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
        op,
        *clients,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    new_scope = scope.merge_scope(pre_executed_scope)
    result = execute_until_in_scope(
        expr,
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
    ).get_value(op, timecontext)
    return result


@trace
def execute_until_in_scope(
    expr,
    scope: Scope,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    clients=None,
    post_execute_=None,
    **kwargs,
) -> Scope:
    """Execute until our op is in `scope`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
    scope : Scope
    timecontext : Optional[TimeContext]
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
    if scope.get_value(op, timecontext) is not None:
        return scope
    if isinstance(op, ops.Literal):
        # special case literals to avoid the overhead of dispatching
        # execute_node
        return Scope(
            {
                op: execute_literal(
                    op, op.value, expr.type(), aggcontext=aggcontext, **kwargs
                )
            },
            timecontext,
        )

    # figure out what arguments we're able to compute on based on the
    # expressions inputs. things like expressions, None, and scalar types are
    # computable whereas ``list``s are not
    computable_args = [arg for arg in op.inputs if is_computable_input(arg)]

    # pre_executed_states is a list of states with same the length of
    # computable_args, these states are passed to each arg
    if timecontext:
        arg_timecontexts = compute_time_context(
            op,
            num_args=len(computable_args),
            timecontext=timecontext,
            clients=clients,
        )
    else:
        arg_timecontexts = [None] * len(computable_args)

    pre_executed_scope = pre_execute(
        op,
        *clients,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )

    new_scope = scope.merge_scope(pre_executed_scope)

    # Short circuit: if pre_execute puts op in scope, then we don't need to
    # execute its computable_args
    if new_scope.get_value(op, timecontext) is not None:
        return new_scope

    # recursively compute each node's arguments until we've changed type.
    # compute_time_context should return with a list with the same length
    # as computable_args, the two lists will be zipping together for
    # further execution
    if len(arg_timecontexts) != len(computable_args):
        raise com.IbisError(
            'arg_timecontexts differ with computable_arg in length '
            f'for type:\n{type(op).__name__}.'
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
        if hasattr(arg, 'op')
        else Scope({arg: arg}, timecontext)
        for (arg, timecontext) in zip(computable_args, arg_timecontexts)
    ]

    # if we're unable to find data then raise an exception
    if not scopes and computable_args:
        raise com.UnboundExpressionError(
            'Unable to find data for expression:\n{}'.format(repr(expr))
        )

    # there should be exactly one dictionary per computable argument
    assert len(computable_args) == len(scopes)

    new_scope = new_scope.merge_scopes(scopes)
    # pass our computed arguments to this node's execute_node implementation
    data = [
        new_scope.get_value(arg.op(), timecontext)
        if hasattr(arg, 'op')
        else arg
        for arg in computable_args
    ]
    result = execute_node(
        op,
        *data,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    computed = post_execute_(op, result, timecontext=timecontext)
    return Scope({op: computed}, timecontext)


execute = Dispatcher('execute')


@execute.register(ir.Expr)
@trace
def main_execute(
    expr,
    params=None,
    scope=None,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    **kwargs,
):
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
    timecontext : Optional[TimeContext]
        timecontext needed for execution
    aggcontext : Optional[ibis.backends.pandas.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    kwargs : Dict[str, object]
        Additional arguments that can potentially be used by individual node
        execution

    Returns
    -------
    result : Union[
        pandas.Series, pandas.DataFrame, ibis.backends.pandas.core.simple_types
    ]

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """

    if scope is None:
        scope = Scope()

    if timecontext is not None:
        # convert timecontext to datetime type, if time strings are provided
        timecontext = canonicalize_context(timecontext)

    if params is None:
        params = {}

    # TODO: make expresions hashable so that we can get rid of these .op()
    # calls everywhere
    params = {k.op() if hasattr(k, 'op') else k: v for k, v in params.items()}
    scope = scope.merge_scope(Scope(params, timecontext))
    return execute_with_scope(
        expr, scope, timecontext=timecontext, aggcontext=aggcontext, **kwargs,
    )


def execute_and_reset(
    expr,
    params=None,
    scope=None,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    **kwargs,
):
    """Execute an expression against data that are bound to it. If no data
    are bound, raise an Exception.

    Notes
    -----
    The difference between this function and :func:`~ibis.backends.pandas.core.execute`
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
    timecontext : Optional[TimeContext]
        timecontext needed for execution
    aggcontext : Optional[ibis.backends.pandas.aggcontext.AggregationContext]
        An object indicating how to compute aggregations. For example,
        a rolling mean needs to be computed differently than the mean of a
        column.
    kwargs : Dict[str, object]
        Additional arguments that can potentially be used by individual node
        execution

    Returns
    -------
    result : Union[
        pandas.Series, pandas.DataFrame, ibis.backends.pandas.core.simple_types
    ]

    Raises
    ------
    ValueError
        * If no data are bound to the input expression
    """
    result = execute(
        expr,
        params=params,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    if isinstance(result, pd.DataFrame):
        schema = expr.schema()
        df = result.reset_index()
        return df.loc[:, schema.names]
    elif isinstance(result, pd.Series):
        return result.reset_index(drop=True)
    return result


compute_time_context = Dispatcher(
    'compute_time_context',
    doc="""\

Compute time context for a node in execution

Notes
-----
For a given node, return with a list of timecontext that are going to be
passed to its children nodes.
time context is useful when data is not uniquely defined by op tree. e.g.
a TableExpr can represent the query select count(a) from table, but the
result of that is different with time context (pd.Timestamp("20190101"),
pd.Timestamp("20200101")) vs (pd.Timestamp("20200101"),
pd.Timestamp("20210101“)), because what data is in "table" also depends on
the time context. And such context may not be global for all nodes. Each
node may have its own context. compute_time_context computes attributes that
are going to be used in executeion and passes these attributes to children
nodes.

Param:
clients: List[ibis.client.Client]
    backends for execution
timecontext : Optional[TimeContext]
    begin and end time context needed for execution

Return:
List[Optional[TimeContext]]
A list of timecontexts for children nodes of the current node. Note that
timecontext are calculated for children nodes of computable args only.
The length of the return list is same of the length of computable inputs.
See ``computable_args`` in ``execute_until_in_scope``
""",
)


@compute_time_context.register(ops.Node)
def compute_time_context_default(
    node, timecontext: Optional[TimeContext] = None, **kwargs
):
    return [timecontext for arg in node.inputs if is_computable_input(arg)]
