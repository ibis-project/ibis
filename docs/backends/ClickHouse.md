---
backend_name: ClickHouse
backend_url: https://clickhouse.yandex/
backend_module: clickhouse
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the ClickHouse backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[clickhouse]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-clickhouse
    ```

{% endfor %}

## Connect

### `ibis.clickhouse.connect`

```python
con = ibis.clickhouse.connect(
    user="username",
    password="password",
    host="hostname",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.clickhouse.connect` is a thin wrapper around [`ibis.backends.clickhouse.Backend.do_connect`][ibis.backends.clickhouse.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.clickhouse.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.clickhouse.connect`, you can also connect to ClickHouse by
passing a properly formatted ClickHouse connection URL to `ibis.connect`

```python
con = ibis.connect(f"clickhouse://{user}:{password}@{host}:{port}?secure={secure}")
```

## ClickHouse playground

ClickHouse provides a free playground with several datasets that you can connect to using `ibis`:

```python
con = ibis.clickhouse.connect(
    host="play.clickhouse.com",
    secure=True,
    user="play",
    password="clickhouse",
)
```

or

```python
con = ibis.connect("clickhouse://play:clickhouse@play.clickhouse.com:443?secure=True")
```
