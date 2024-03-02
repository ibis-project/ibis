# Ibis

[![Documentation status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://ibis-project.org)
[![Project chat](https://img.shields.io/badge/zulip-join_chat-purple.svg?logo=zulip)](https://ibis-project.zulipchat.com)
[![Anaconda badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)
[![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml?query=branch%3Amain)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml?query=branch%3Amain)
[![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/main.svg)](https://codecov.io/gh/ibis-project/ibis)

## What is Ibis?

Ibis is the portable Python dataframe library:

- Fast local dataframes with the default DuckDB backend
- Lazy (by default) or interactive execution
- [Mix Python dataframe and SQL code](#python--sql-better-together)
- Use the same dataframe API for [20+ backends](#backends)
- Iterate locally and deploy remotely by [changing a single line of code](#portability)

See the documentation on ["Why Ibis?"](https://ibis-project.org/why) to learn more.

## Getting started

You can `pip install` Ibis with a backend and example data:

```bash
pip install 'ibis-framework[duckdb,examples]'
```

> [!TIP]
> See the [installation guide](https://ibis-project.org/install) for more installation options.

Then use Ibis:

```python
>>> import ibis
>>> ibis.options.interactive = True
>>> t = ibis.examples.penguins.fetch()
>>> t
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │  2007 │
│ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │        3800 │ female │  2007 │
│ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │        3250 │ female │  2007 │
│ Adelie  │ Torgersen │           NULL │          NULL │              NULL │        NULL │ NULL   │  2007 │
│ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │        3450 │ female │  2007 │
│ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │        3650 │ male   │  2007 │
│ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │        3625 │ female │  2007 │
│ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │        4675 │ male   │  2007 │
│ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │        3475 │ NULL   │  2007 │
│ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │        4250 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
>>> g = t.group_by(["species", "island"]).agg(count=t.count()).order_by("count")
>>> g
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species   ┃ island    ┃ count ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string    │ string    │ int64 │
├───────────┼───────────┼───────┤
│ Adelie    │ Biscoe    │    44 │
│ Adelie    │ Torgersen │    52 │
│ Adelie    │ Dream     │    56 │
│ Chinstrap │ Dream     │    68 │
│ Gentoo    │ Biscoe    │   124 │
└───────────┴───────────┴───────┘
```

> [!TIP]
> See the [getting started tutorial](https://ibis-project.org/getting-started) for a full introduction to Ibis.

## Python + SQL: better together

For most backends, Ibis works by compiling its dataframe expressions into SQL:

```python
>>> ibis.to_sql(g)
SELECT
  "t1"."species",
  "t1"."island",
  "t1"."count"
FROM (
  SELECT
    "t0"."species",
    "t0"."island",
    COUNT(*) AS "count"
  FROM "penguins" AS "t0"
  GROUP BY
    1,
    2
) AS "t1"
ORDER BY
  "t1"."count" ASC
```

You can mix SQL and Python code:

```python
>>> t1 = t.sql("SELECT species, island, count(*) AS count FROM penguins GROUP BY 1, 2")
>>> t1
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species   ┃ island    ┃ count ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string    │ string    │ int64 │
├───────────┼───────────┼───────┤
│ Adelie    │ Torgersen │    52 │
│ Adelie    │ Biscoe    │    44 │
│ Adelie    │ Dream     │    56 │
│ Gentoo    │ Biscoe    │   124 │
│ Chinstrap │ Dream     │    68 │
└───────────┴───────────┴───────┘
>>> t2 = t1.order_by("count")
>>> t2
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species   ┃ island    ┃ count ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string    │ string    │ int64 │
├───────────┼───────────┼───────┤
│ Adelie    │ Biscoe    │    44 │
│ Adelie    │ Torgersen │    52 │
│ Adelie    │ Dream     │    56 │
│ Chinstrap │ Dream     │    68 │
│ Gentoo    │ Biscoe    │   124 │
└───────────┴───────────┴───────┘
```

This allows you to combine the flexibility of Python with the scale and performance of modern SQL OLAP engines.

## Backends

Ibis supports 20+ backends:

- [Apache Arrow DataFusion](https://ibis-project.org/backends/datafusion/)
- [Apache Druid](https://ibis-project.org/backends/druid/)
- [Apache Flink](https://ibis-project.org/backends/flink)
- [Apache Impala](https://ibis-project.org/backends/impala/)
- [Apache PySpark](https://ibis-project.org/backends/pyspark/)
- [BigQuery](https://ibis-project.org/backends/bigquery/)
- [ClickHouse](https://ibis-project.org/backends/clickhouse/)
- [Dask](https://ibis-project.org/backends/dask/)
- [DuckDB](https://ibis-project.org/backends/duckdb/)
- [Exasol](https://ibis-project.org/backends/exasol)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)
- [MySQL](https://ibis-project.org/backends/mysql/)
- [Oracle](https://ibis-project.org/backends/oracle/)
- [Pandas](https://ibis-project.org/backends/pandas/)
- [Polars](https://ibis-project.org/backends/polars/)
- [PostgreSQL](https://ibis-project.org/backends/postgresql/)
- [SQL Server](https://ibis-project.org/backends/mssql/)
- [SQLite](https://ibis-project.org/backends/sqlite/)
- [Snowflake](https://ibis-project.org/backends/snowflake)
- [Trino](https://ibis-project.org/backends/trino/)

## Portability

You can set the backend Ibis uses:

```python
>>> ibis.set_backend("duckdb")
>>> ibis.set_backend("polars")
>>> ibis.set_backend("datafusion")
```

More conventionally, you'll create a connection object:

```python
>>> con = ibis.duckdb.connect()
>>> con = ibis.polars.connect()
>>> con = ibis.datafusion.connect()
```

And work with tables in that backend:

```python
>>> t = con.table("penguins")
```

Or, for most backends, read from common file formats like CSV or Parquet:

```python
>>> t = con.read_csv("penguins.csv")
>>> t = con.read_parquet("penguins.parquet")
```

By changing your connection object, you can work with any backend Ibis supports using the same API.

## Community and contributing

Ibis is an open source project and welcomes contributions from anyone in the community.

- Read [the contributing guide](https://github.com/ibis-project/ibis/blob/main/docs/CONTRIBUTING.md).
- We care about keeping the community welcoming for all. Check out [the code of conduct](https://github.com/ibis-project/ibis/blob/main/docs/CODE_OF_CONDUCT.md).
- The Ibis project is open sourced under the [Apache License](https://github.com/ibis-project/ibis/blob/main/LICENSE.txt).

Join our community by interacting on GitHub or chatting with us on [Zulip](https://ibis-project.zulipchat.com/).

For more information visit https://ibis-project.org/.
