# Roadmap

This document is an outline of the next set of major efforts within ibis.

## Make SQL and SQLAlchemy base backends independent

At the moment, Ibis has two different ways of implementing SQL-based backends,
with SQLAlchemy, and with a custom SQL compiler. For historical reasons, the
SQLAlchemy backend was built subclassing the custom Ibis SQL compiler. But
conceptually, this doesn't make sense. The right way to have the code structured
would be to have a base backend with the code shared among both backends (and
possibly other non-SQL backends). And then have both base backends subclass that.

## Move SQL base backend, and backends based on it to external repositories

Ideally, for SQL backends, Ibis should provide one single way to generate the
SQL. Using SQLAlchemy. For historical reasons, Ibis implemented its custom
SQL generator/compiler. But this implementation is basically a suboptimal copy
of SQLAlchemy, and it doesn't add value, but it does add a lot of code to
maintain and complexity. In the long-term, the ideal status would be to have
all SQL backends implemented based on SQLAlchemy. It will take some time
to have the current Ibis compiler based backends moved to SQLAlchemy. But to
make this transition easy, would be good to remove these backends (including
the base SQL backend) in separate projects. So, new backends based on SQLAlchemy
can be developed in parallel, without conflicts or extra complexity, and
users can decide which one to use for a given backend.

## Standardize UDFs (User Defined Functions)

A few backends have support for UDFs. Impala, Pandas and BigQuery all have at
least some level of support for user-defined functions. This mechanism should
be extended to other backends where possible. We outline different approaches
to adding UDFs to the backends that are well-supported but currently do not
have a UDF implementation. Development of a standard interface for UDFs is
ideal, so that it’s easy for new backends to implement the interface.
