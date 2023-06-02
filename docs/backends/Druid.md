---
backend_name: Druid
backend_url: https://druid.apache.org/
backend_module: druid
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

{% include 'backends/badges.md' %}

!!! experimental "Introduced in v5.0"

    The Druid backend is experimental and is subject to backwards incompatible changes.

## Install

Install `ibis` and dependencies for the Druid backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[druid]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-druid
    ```

{% endfor %}

## Connect

### `ibis.druid.connect`

```python
con = ibis.druid.connect(
    host="hostname",
    port=8082,
    database="druid/v2/sql",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.druid.connect` is a thin wrapper around [`ibis.backends.druid.Backend.do_connect`][ibis.backends.druid.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.druid.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

### `ibis.connect` URL format

In addition to `ibis.druid.connect`, you can also connect to Druid by
passing a properly formatted Druid connection URL to `ibis.connect`

```python
con = ibis.connect("druid://localhost:8082/druid/v2/sql")
```
