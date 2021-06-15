# Roadmap

This document is an outline of the next set of major efforts within ibis.

### Standardize UDFs (User Defined Functions)

A few backends have support for UDFs. Impala, Pandas and BigQuery all have at
least some level of support for user-defined functions. This mechanism should
be extended to other backends where possible. We outline different approaches
to adding UDFs to the backends that are well-supported but currently do not
have a UDF implementation. Development of a standard interface for UDFs is
ideal, so that it’s easy for new backends to implement the interface.
