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

## Connecting to older Oracle databases

`ibis` uses the `python-oracledb` "thin client" to connect to Oracle databases.
Because early versions of Oracle did not perform case-sensitive checks in
passwords, some DBAs disable case sensitivity to avoid requiring users to update
their passwords. If case-sensitive passwords are disabled, then Ibis will not be
able to connect to the database.

To check if case-sensitivity is enforced you can run

```sql
show parameter sec_case_sensitive_logon;
```

If the returned value is `FALSE` then Ibis will _not_ connect.

For more information, see this [issue](https://github.com/oracle/python-oracledb/issues/26).
