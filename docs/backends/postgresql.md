---
backend_name: PostgreSQL
backend_url: https://www.postgresql.org/
backend_module: postgres
backend_param_style: a SQLAlchemy-style URI
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

# PostgreSQL

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the Postgres backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[postgres]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-postgres
    ```

{% endfor %}

## Connect

### `ibis.postgres.connect`

```python
con = ibis.postgres.connect(
    user="username",
    password="password",
    host="hostname",
    port=5432,
    database="database",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.postgres.connect` is a thin wrapper around [`ibis.backends.postgres.Backend.do_connect`][ibis.backends.postgres.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.postgres.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.postgres.connect`, you can also connect to Postgres by
passing a properly formatted Postgres connection URL to `ibis.connect`

```python
con = ibis.connect(f"postgres://{user}:{password}@{host}:{port}/{database}")
```
