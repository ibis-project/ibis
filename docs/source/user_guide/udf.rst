.. _udf:

User Defined Functions
======================

Ibis provides a mechanism for writing custom scalar and aggregate functions,
with varying levels of support for different backends. UDFs/UDAFs are a complex
topic.

This section of the documentation will discuss some of the backend specific
details of user defined functions.

.. warning::

   The UDF API is provisional and subject to change.

The next backends provide UDF support:

- :ref:`udf.impala`
- :ref:`udf.pandas`
- :ref:`udf.bigquery`
