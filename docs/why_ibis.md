---
hide:
  - footer
---

# Why try Ibis?

Ibis is a dataframe interface to execution engines with support for [15+
backends](./backends/index.md). Ibis doesn't replace your existing execution
engine, it _extends_ it with powerful abstractions and intuitive syntax.

Ibis works with what you already have, so why not check out our [getting started
guide](./getting_started.md)?

# How does Ibis compare to...

Let us stop you there. Ibis is an interface that empowers you to craft complex
and powerful queries that execute on your engine of choice.

The answer to, "how does Ibis compare to `X`?" is "Ibis helps you use `X`."

!!! tip "[Get in touch](https://github.com/ibis-project/ibis/issues) if you're having trouble using `X`!"

Now that we've said that, here are some other tools that you might compare Ibis
with:

## Big Data engines like `BigQuery`, `Snowflake`, `Spark`, ...

See above. Ibis works with your existing execution engine, it doesn't replace it.

## SQL

SQL is the 800 lb gorilla in the room. One of our developers [gave a whole
talk](https://www.youtube.com/watch?v=XdZklxTbCEA) comparing Ibis and SQL, but
we can summarize some key points:

- SQL fails at runtime, Ibis validates expressions as you construct them
- Ibis is written in Python and features some pretty killer tab-completion
- Ibis lets you use SQL when you want to (for our SQL-based backends)

If your SQL-fu is strong, we might not convince you to leave it all behind, but
check out our [Ibis for SQL Programmers guide](./ibis-for-sql-programmers.ipynb)
and see if it whets your appetite.

## `pandas`

`pandas` is the 800 lb panda in the room. Ibis, like every dataframe API in the
PyData ecosystem, takes a fair bit of inspiration from `pandas`.

And like the other engine comparisons above, Ibis doesn't replace `pandas`, it works _with_ `pandas`.

`pandas` is an in-memory analysis engine -- if your data are bigger than the
amount of RAM you have, things will go poorly.

Ibis defers execution, and is agnostic to the backend that runs a given query.
If your analysis is causing `pandas` to hit an out-of-memory error, you can use
Ibis to quickly and easily switch to a different backend that supports
out-of-core execution.

Ibis syntax is similar to `pandas` syntax, but it isn't a drop-in replacement.
Check out our [Ibis for Pandas Users guide](./ibis-for-pandas-users.ipynb) if
you'd like to give Ibis a try!

## `sqlalchemy` and `sqlglot`

[`sqlalchemy`](https://www.sqlalchemy.org/) and
[`sqlglot`](https://sqlglot.com/sqlglot.html) are amazing tools and we are big
fans. Ibis uses both of these heavily to validate and generate SQL to send to
our SQL backends.

If you need super-fine-grained control over which SQL primitives are used to
construct a query and you are using Python, SQLAlchemy is definitely the tool
for you.

If you are looking for a Python-based SQL transpiler, we strongly recommend
using SQLGlot.

If you are looking for a dataframe API to construct and execute your analytics
queries against a large collection of powerful execution engines, then allow us
point you at the [Ibis Getting Started guide](./getting_started.md).
