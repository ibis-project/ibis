.. currentmodule:: ibis
.. _impala-udf:

*************************
Using Impala UDFs in Ibis
*************************

Impala currently supports user-defined scalar functions (known henceforth as
*UDFs*) and aggregate functions (respectively *UDAs*) via a C++ extension API.

Initial support for using C++ UDFs in Ibis came in version 0.4.0.

Using scalar functions (UDFs)
-----------------------------

Let's take an example to illustrate how to make a C++ UDF available to
Ibis. Here is a function that computes an approximate equality between floating
point values:

.. code-block:: c++

   #include "impala_udf/udf.h"

   #include <cctype>
   #include <cmath>

   BooleanVal FuzzyEquals(FunctionContext* ctx, const DoubleVal& x, const DoubleVal& y) {
     const double EPSILON = 0.000001f;
     if (x.is_null || y.is_null) return BooleanVal::null();
     double delta = fabs(x.val - y.val);
     return BooleanVal(delta < EPSILON);
   }

You can compile this to either a shared library (a ``.so`` file) or to LLVM
bitcode with clang (a ``.ll`` file). Skipping that step for now (will add some
more detailed instructions here later, promise).

To make this function callable, we first create a UDF wrapper with
``ibis.impala.wrap_udf``:

.. code-block:: python

   library = '/ibis/udfs/udftest.ll'
   inputs = ['double', 'double']
   output = 'boolean'
   symbol = 'FuzzyEquals'
   udf_db = 'ibis_testing'
   udf_name = 'fuzzy_equals'

   wrapper = ibis.impala.wrap_udf(library, inputs, output, symbol, name=udf_name)

In typical workflows, you will set up a UDF in Impala once then use it
thenceforth. So the *first time* you do this, you need to create the UDF in
Impala:

.. code-block:: python

   client.create_udf(wrapper, name=udf_name, database=udf_db)

Now, we must register this function as a new Impala operation in Ibis. This
must take place each time you load your Ibis session.

.. code-block:: python

   operation_class = wrapper.to_operation()
   ibis.impala.add_operation(operation_class, udf_name, udf_db)

Lastly, we define a *user API* to make ``fuzzy_equals`` callable on Ibis
expressions:

.. code-block:: python

   def fuzzy_equals(left, right):
       """
       Approximate equals UDF

       Parameters
       ----------
       left : numeric
       right : numeric

       Returns
       -------
       is_approx_equal : boolean
       """
       op = operation_class(left, right)
       return op.to_expr()

Now, we have a callable Python function that works with Ibis expressions:

.. code-block:: python

   In [35]: db = c.database('ibis_testing')

   In [36]: t = db.functional_alltypes

   In [37]: expr = fuzzy_equals(t.float_col, t.double_col / 10)

   In [38]: expr.execute()[:10]
   Out[38]:
   0     True
   1    False
   2    False
   3    False
   4    False
   5    False
   6    False
   7    False
   8    False
   9    False
   Name: tmp, dtype: bool

Note that the call to ``ibis.impala.add_operation`` must happen each time you
use Ibis. If you have a lot of UDFs, I suggest you create a file with all of
your wrapper declarations and user APIs that you load with your Ibis session to
plug in all your own functions.

Using aggregate functions (UDAs)
--------------------------------

Coming soon.

Adding UDF functions to Ibis types
----------------------------------

Coming soon.

Installing the Impala UDF SDK on OS X and Linux
-----------------------------------------------

Coming soon.

Impala types to Ibis types
--------------------------

Coming soon. See ``ibis.schema`` for now.
