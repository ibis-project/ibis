---
backend_name: Trino
backend_url: https://trino.io
backend_module: trino
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
---

# Trino

{% include 'backends/badges.md' %}

!!! experimental "Introduced in v4.0"

    The Trino backend is experimental and is subject to backwards incompatible changes.

## Install

Install `ibis` and dependencies for the Trino backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[trino]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-trino
    ```

{% endfor %}

## Connect

### `ibis.trino.connect`

```python
con = ibis.trino.connect(
    user="user",
    password="password",
    port=8080,
    database="database",
    schema="default",
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.trino.connect` is a thin wrapper around [`ibis.backends.trino.Backend.do_connect`][ibis.backends.trino.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.trino.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->
