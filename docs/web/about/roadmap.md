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

## Simplify development of SQLAlchemy backends

Given a SQLAlchemy engine, Ibis should be able to have most of the information
it needs to work with that engine. There are cases where backends need to
extend SQLAlchemy engines, such as with non-standard operations, custom types,
UDFs or GIS functionality. Other than those cases, for simple queries, Ibis
should be able to simply work given a SQLAlchemy engine. We should make Ibis
work with arbitrary engines without implementing a backend. And for engines
that are extended, it should be possible to write backends in the simplest and
fastest way.

## Execution output

Currently, the output of executing an expression (i.e. `expr.execute()`) is
returned as a pandas DataFrame. This is convenient, and the output can
be further processed with the extensive features of pandas, or saved to
different formats, with the pandas exporting interfaces. But in some cases,
exporting to pandas implies adding pandas as an undesired dependency, or
having an extra overheat. Often, Ibis backends are implemented using a
[Python DB API 2.0](https://www.python.org/dev/peps/pep-0249/) interface,
and accessing its Cursor object can be more convenient. In other cases,
serializing from the backend to Apache Arrow is possible (and efficient)
and accessing in this format would be the preferred way. Ibis should provide
a more flexible way to access the results. For example, via an Ibis option,
an argument (e.g. `.execute(format='cursor')`), different methods for
different formats, or another way.

## Base pandas-API backend

There are currently backends for pandas and Dask, which share a very similar
API. Other projects also implement the same API based on pandas, such as
Vaex, Modin, Koalas. The current backends (pandas and Dask) have a significant
amount of duplication. And implementing more backends based on the pandas API
require the same duplication, and a big effort. It should be possible to
create a base backend where Ibis operations are mapped to pandas operations,
and that backends for pandas-like backends can reuse, and don't need to
reimplement.

## Improve API for file psuedo-backends

Ibis currently supports loading from files to the pandas backend for few formats,
csv, hdf5 and parquet. Those are implemented as backends, while they actually
use the pandas backend under the hood. Besides loading the data into the pandas
backend, those pseudo-backends provides functionalities to list files as if
they were tables, and to save the data back to files. While the current approach
kind of makes sense, it's misleading, as in the future Ibis could implement
actual file base backends. For example, a csv backend that for a filter
expression, reads the csv file ignoring the filtered lines. So, the current
backends create the wrong expectations, and make the concept of backend
confusing. Implementing the current functionality as a different concept,
like `FileManager` instead of backend, that interacts with the pandas backend
(or any other backend), would make things clearer. Also, it could possibly
be implemented in a way that different formats are supported at the same
time (like dealing with a directory with a mix of csv and parquet files).

## Standardize UDFs (User Defined Functions)

A few backends have support for UDFs. Impala, Pandas and BigQuery all have at
least some level of support for user-defined functions. This mechanism should
be extended to other backends where possible. We outline different approaches
to adding UDFs to the backends that are well-supported but currently do not
have a UDF implementation. Development of a standard interface for UDFs is
ideal, so that it’s easy for new backends to implement the interface.

## Better integration of the website and the docs

We are using Sphinx for the docs, and Pysuerga for the website. This makes
sense, since Sphinx work well for documentation (automatic API docs,
cross-referencing of pages,...), and Pysuerga works well in generating
a website in a very simple way. But the experience for the user is that
the two are different, with different look and feel, and with just one
link from the website to the docs. While internally keeping what we've
got, we can use the Pysuerga base template as the base template for
Sphinx, and keep the same styles, the same navigation bar and the footer
in all pages of the docs. The experience for the users will be much
better, as well as the navigation, and the whole site will feel a single
one.
