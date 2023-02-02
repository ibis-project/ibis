# Ibis v4.0.0

**by Patrick Clarke**

09 January 2023

## Introduction

Ibis 4.0 has officially been released as the latest version of the package.
This release includes several new backends, improved functionality, and some major internal refactors.
A full list of the changes can be found in the [project release notes](../release_notes.md).
Let’s talk about some of the new changes 4.0 brings for Ibis users.

## Backends

Ibis 4.0 brings [Polars](https://ibis-project.org/backends/Polars/), [Snowflake](https://ibis-project.org/backends/Snowflake/), and [Trino](https://ibis-project.org/backends/Trino/) into an already-impressive stock of supported backends.
The [Polars](https://www.pola.rs/) backend adds another way for users to work locally with DataFrames.
The [Snowflake](https://www.snowflake.com/en/) and [Trino](https://trino.io/) backends add a free and familiar python API to popular data warehouses.

Alongside these new backends, Google BigQuery and Microsoft SQL have been moved to the main repo, so their release cycle will follow the Ibis core.

## Functionality

There are a lot of improvements incoming, but some notable changes include:

- [read API](https://github.com/ibis-project/ibis/pull/5005): allows users to read various file formats directly into their [configured `default_backend`](https://ibis-project.org/api/config/?h=default#ibis.config.Options) (default DuckDB) through `read_*` functions, which makes working with local files easier than ever.
- [to_pyarrow and to_pyarrow_batches](https://github.com/ibis-project/ibis/pull/4454#issuecomment-1262640204): users can now return PyArrow objects (Tables, Arrays, Scalars, RecordBatchReader) and therefore grants all of the functionality that PyArrow provides
- [JSON getitem](https://github.com/ibis-project/ibis/pull/4525): users can now run getitem on a JSON field using Ibis expressions with some backends
- [Plotting support through `__array__`](https://github.com/ibis-project/ibis/pull/4547): allows users to plot Ibis expressions out of the box

## Refactors

This won't be visible to most users, but the project underwent a series of refactors that spans [multiple PRs](https://github.com/ibis-project/ibis/pulls?q=is%3Apr+is%3Amerged+%22refactor%3A%22+milestone%3A4.0.0).
Notable changes include removing intermediate expressions, improving the testing framework, and UX updates.

## Additional Changes

As mentioned previously, additional functionality, bugfixes, and more have been included in the latest 4.0 release.
To stay up to date and learn more about recent changes: check out the project's homepage at [ibis-project.org](https://ibis-project.org/docs), follow [@IbisData](https://twitter.com/IbisData) on Twitter, find the source code and community on [GitHub](https://github.com/ibis-project/ibis), and join the discussion on [Gitter](https://gitter.im/ibis-dev/Lobby).

As always, try Ibis by [installing](https://ibis-project.org/install/) it today.
