"""The dask backend is a very close port of the pandas backend, and thus
has the same caveats.

The dask backend is a departure from the typical ibis backend in that it
doesn't compile to anything, and the execution of the ibis expression
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
This is a no-op because we execute ibis expressions directly. However, the
dask backend will pass back a dask expression that you can run `.compute()`
on to evaluate it.

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

Defining a ``post_execute`` rule for :class:`~ibis.expr.operations.WindowOp`
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

See ibis.common.scope for details about the implementaion.
"""

from __future__ import absolute_import

import dask.dataframe as dd

from ibis.backends.pandas.core import (  # noqa: F401
    boolean_types,
    compute_time_context,
    compute_time_context_default,
    date_types,
    execute,
    execute_and_reset,
    execute_until_in_scope,
    execute_with_scope,
    fixed_width_types,
    floating_types,
    integer_types,
    is_computable_input,
    is_computable_input_arg,
    main_execute,
    numeric_types,
    temporal_types,
    time_types,
    timedelta_types,
)

from .dispatch import (  # noqa: F401
    execute_literal,
    execute_node,
    post_execute,
    pre_execute,
)
from .trace import trace  # noqa: F401

is_computable_input.register(dd.core.Scalar)(is_computable_input_arg)
