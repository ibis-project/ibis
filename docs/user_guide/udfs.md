# User Defined Functions

!!! experimental "UD(A)Fs are unstable"

    The user-defined elementwise and aggregate function APIs are provisional
    and subject to change.

Ibis has mechanisms for writing custom scalar and aggregate functions with
varying levels of support for depending on the backend

User-defined function are a complex and interesting topic. Please get involved
if you're interested in working on them!

The following backends provide some level of support for user-defined functions:

- [Google BigQuery](https://github.com/ibis-project/ibis-bigquery)
- [Pandas](../backends/Pandas.md)
- [PostgreSQL](../backends/PostgreSQL.md)
- [Datafusion](../backends/Datafusion.md)
- [Impala](../backends/Impala.md)
