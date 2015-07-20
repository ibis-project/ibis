.. _faq:

**************************
Frequently Asked Questions
**************************

Ibis and other Python projects
------------------------------


Ibis and Spark / PySpark
------------------------

Ibis sits at a different layer of the cake, so to speak. Spark is a distributed
computing engine with multiple language APIs; Ibis is a Python-centric library
enabling seamless analytical access to big data. In the same way that Ibis uses
Impala, Spark could also be used as an execution engine for Ibis.

The Spark Python API (PySpark) enables Python programmers to access the Spark
computation model and ecosystem of libraries. Spark itself is an execution
engine and set of analytics libraries that provide a higher performance, more
scalable alternative to Hadoop MapReduce for many applications.

In future releases, Ibis will take advantage of architectural features in the
upcoming Impala roadmap, specifically external user-defined functions operating
on shared memory without data serialization overhead and a canonical in-memory
columnar data layout. This will enable the Python community to utilize existing
libraries and high performance computing tools (e.g. wrapping C/C++) code
without suffering slowdown from data marshalling. Python's scientific computing
tools, like NumPy, will be easy to integrate with Impala's byte-level data
structures. We will also leverage LLVM runtime code generation to achieve
native hardware performance inside Impala.

That being said, Spark and its rich ecosystem would be extremely useful to
expose to Ibis users through an seamlessly integrated user
experience. Workflows involving multiple systems (for example: using Spark
MLLib alongside Impala) would be straightforward to put together with HDFS as
the common point of contact. In such cases, Ibis would use PySpark as a
dependency. This of course includes using Spark's Dataframe APIs where relevant
to translate Ibis data operations into Spark operations.
