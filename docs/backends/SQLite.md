---
backend_name: SQLite
backend_url: https://www.sqlite.org/
backend_module: sqlite
imports: ["CSV", "Parquet", "JSON", "PyArrow", "Pandas", "SQLite", "Postgres"]
---

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the SQLite backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[sqlite]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-sqlite
    ```

{% endfor %}

## Connect

### `ibis.sqlite.connect`

```python
con = ibis.sqlite.connect()  # (1)
```

1. Use an ephemeral, in-memory database

```python
con = ibis.sqlite.connect("mydb.sqlite")  # (1)
```

1. Connect to, or create, a local SQLite file

<!-- prettier-ignore-start -->
!!! info "`ibis.sqlite.connect` is a thin wrapper around [`ibis.backends.sqlite.Backend.do_connect`][ibis.backends.sqlite.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.sqlite.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.sqlite.connect`, you can also connect to SQLite by
passing a properly formatted SQLite connection URL to `ibis.connect`

```python
con = ibis.connect("sqlite:///path/to/local/file")
```

```python
con = ibis.connect("sqlite://") # (1)
```

1. ephemeral, in-memory database
