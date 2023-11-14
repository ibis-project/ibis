---
backend_name: MySQL
backend_url: https://www.mysql.com/
backend_module: mysql
backend_param_style: a SQLAlchemy-style URI
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

# MySQL

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the MySQL backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[mysql]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-mysql
    ```

{% endfor %}

## Connect

### `ibis.mysql.connect`

```python
con = ibis.mysql.connect(
    user="username",
    password="password",
    host="hostname",
    port=3306,
    database="database",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.mysql.connect` is a thin wrapper around [`ibis.backends.mysql.Backend.do_connect`][ibis.backends.mysql.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.mysql.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.mysql.connect`, you can also connect to MySQL by
passing a properly formatted MySQL connection URL to `ibis.connect`

```python
con = ibis.connect(f"mysql://{user}:{password}@{host}:{port}/{database}")
```
