---
backend_name: DuckDB
backend_url: https://duckdb.org/
backend_module: duckdb
exports: ["PyArrow", "Parquet", "Delta Lake", "CSV", "Pandas"]
imports:
  [
    "CSV",
    "Parquet",
    "Delta Lake",
    "JSON",
    "PyArrow",
    "Pandas",
    "SQLite",
    "Postgres",
  ]
---

# DuckDB

{% include 'backends/badges.md' %}

??? danger "`duckdb` >= 0.5.0 requires `duckdb-engine` >= 0.6.2"

      If you encounter problems when using `duckdb` >= **0.5.0** you may need to
      upgrade `duckdb-engine` to at least version **0.6.2**.

      See [this issue](https://github.com/ibis-project/ibis/issues/4503) for
      more details.

## Install

Install `ibis` and dependencies for the DuckDB backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[duckdb]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-duckdb
    ```

{% endfor %}

## Connect

### `ibis.duckdb.connect`

```python
con = ibis.duckdb.connect()  # (1)
```

1. Use an ephemeral, in-memory database

```python
con = ibis.duckdb.connect("mydb.duckdb")  # (1)
```

1. Connect to, or create, a local DuckDB file

<!-- prettier-ignore-start -->
!!! info "`ibis.duckdb.connect` is a thin wrapper around [`ibis.backends.duckdb.Backend.do_connect`][ibis.backends.duckdb.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.duckdb.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.duckdb.connect`, you can also connect to DuckDB by
passing a properly formatted DuckDB connection URL to `ibis.connect`

```python
con = ibis.connect("duckdb:///path/to/local/file")
```

```python
con = ibis.connect("duckdb://") # (1)
```

1. ephemeral, in-memory database

## MotherDuck

The DuckDB backend supports [MotherDuck](https://motherduck.com). If you have an
account, you can connect to MotherDuck by passing in the string `md:` or
`motherduck:`. `ibis` will trigger the authentication prompt in-browser.

```python
>>> import ibis

>>> con = ibis.duckdb.connect("md:")
```

<!-- prettier-ignore-start -->
!!! info "Authentication to MotherDuck will trigger on the first call that requires retrieving information (in this case `list_tables`)"
<!-- prettier-ignore-end -->

```python
>>> con.list_tables()
Attempting to automatically open the SSO authorization page in your default browser.
1. Please open this link to login into your account: https://auth.motherduck.com/activate
2. Enter the following code: ZSRQ-GJQS


Token successfully retrieved ✅
You can store it as an environment variable to avoid having to log in again:
  $ export motherduck_token='****************'

['penguins']
```

## File Support

<!-- prettier-ignore-start -->
::: ibis.backends.duckdb.Backend.read_csv
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_parquet
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_delta
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_json
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_in_memory
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_sqlite
    options:
      heading_level: 4
      show_docstring_examples: false
      show_docstring_returns: false
::: ibis.backends.duckdb.Backend.read_postgres
    options:
      heading_level: 4
      show_docstring_returns: false
<!-- prettier-ignore-end -->
