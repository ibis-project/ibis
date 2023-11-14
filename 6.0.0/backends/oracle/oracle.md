---
backend_name: Oracle
backend_url: https://docs.oracle.com/en/database/oracle/oracle-database/index.html
backend_module: oracle
backend_param_style: a SQLAlchemy connection string
backend_connection_example: ibis.connect("oracle://user:pass@host:port/service_name")
is_experimental: true
version_added: "6.0"
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

# Oracle

{% include 'backends/badges.md' %}

!!! experimental "Introduced in v6.0"

    The Oracle backend is experimental and is subject to backwards incompatible changes.

## Install

Install `ibis` and dependencies for the Oracle backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[oracle]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-oracle
    ```

{% endfor %}

## Connect

### `ibis.oracle.connect`

```python
con = ibis.oracle.connect(
    user="username",
    password="password",
    host="hostname",
    port=1521,
    database="database",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.oracle.connect` is a thin wrapper around [`ibis.backends.oracle.Backend.do_connect`][ibis.backends.oracle.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.oracle.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.oracle.connect`, you can also connect to Oracle by
passing a properly formatted Oracle connection URL to `ibis.connect`

```python
con = ibis.connect(f"oracle://{user}:{password}@{host}:{port}/{database}")
```
