.. _roadmap:

Roadmap
=======
This document is an outline of the next set of major efforts within ibis.


Long Term Goals
---------------
This section outlines broader, longer-term goals for the project alongside a
few short-term goals and provides information and direction for a few key areas
of focus over the next 1-2 years, possibly longer depending on the amount of
time the developers of Ibis can devote to the project.

Compiler Structure
~~~~~~~~~~~~~~~~~~
Separation of Concerns
^^^^^^^^^^^^^^^^^^^^^^
The current architecture of the ibis compiler has a few key problems that need
to be addressed to ensure longevity and maintainability of the project going
forward.

The main structural problem is that there isn’t one place where expression
optimizations and transformations happen. Sometimes optimizations occur as an
expression is being built, and other times they occur on a whole expression.

The solution is to separate the expression construction and optimization into
two phases, and guarantee that a constructed expression, if compiled without
optimization, maps clearly to SQL constructs.

The optimization pass would happen just before compilation and would be free to
optimize whole expression trees.

This approach lets us optimize queries piece by piece, as opposed to having to
provide all optimization implementations in a single pull request.

Unifying Table and Column Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Right now, it is very difficult to customize the way the operations underlying
table expressions are compiled. The logic to compile them is hard-coded in each
backend (or the compiler’s parent class). This needs to be addressed, if only
to ease the burden of implementing the UNNEST operation and make the codebase
easier to understand and maintain.

Depth
~~~~~
Backend-Specific Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
As the number of ibis users and use cases grows there will be an increasing
need for individual backends to support more exotic operations. Many SQL
databases have features that are unique only to themselves and often this is
why people will choose that technology over another. Ibis should support an API
that reflects the backend that underlies an expression and expose the
functionality of that specific backend.

A concrete example of this is the `FARM_FINGERPRINT <>`_ function in BigQuery.

It is unlikely that the main ValueExpr API will ever grow such a method, but a
BigQuery user shouldn’t be restricted to using only the methods this API
provides. Moreover, users should be able to bring their own methods to this API
without having to consult the ibis developers and without the addition of such
operations polluting the namespace of the main API.

One drawback to enabling this is that it provides an incentive for people to
define operations with a backend-specific spelling (presumably in the name of
expediency) that may actually be easily generalizable to or useful for other
backends. This behavior should be discouraged to the extent possible.

Standardize UDFs (User Defined Functions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A few backends have support for UDFs. Impala, Pandas and BigQuery all have at
least some level of support for user-defined functions. This mechanism should
be extended to other backends where possible. We outline different approaches
to adding UDFs to the backends that are well-supported but currently do not
have a UDF implementation. Development of a standard interface for UDFs is
ideal, so that it’s easy for new backends to implement the interface.

Breadth
~~~~~~~
The major breadth-related question ibis is facing is how to grow the number of
backends in ibis in a scalable, minimum-maintenance way is an open question.

Currently there is a test suite that runs across all backends that xfails tests
when a backend doesn’t implement a particular operation.

At minimum we need a way to display which backends implement which operations.
With the ability to provide custom operations we also need a way to display the
custom operations that each backend provides.

Backend-Specific Goals
----------------------
These are goals related to specific backends

Pandas
~~~~~~
Speed up grouped, rolling, and simple aggregations using numba
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pandas aggregations are quite slow relative to an equivalent numba
implementation, for various reasons. Since ibis hides the implementation
details of a particular expression we can experiment with using different
aggregation implementations.

Dask
~~~~
Implement a Dask backend
^^^^^^^^^^^^^^^^^^^^^^^^
There is currently no way in ibis to easily parallelize a computation on a
single machine, let alone distribute a computation across machines.

Dask provides APIs for doing such things.

Spark
~~~~~
Implement a SparkSQL backend

SparkSQL provides a way to execute distributed SQL queries similar to other
backends supported by ibis such as Impala and BigQuery.
