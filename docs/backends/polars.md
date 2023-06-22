---
backend_name: Polars
backend_url: https://pola-rs.github.io/polars-book/user-guide/index.html
backend_module: polars
is_experimental: true
version_added: "4.0"
exports: ["PyArrow", "Parquet", "Delta Lake", "CSV", "Pandas"]
imports: ["CSV", "Parquet", "Delta Lake", "Pandas"]
---

# Polars

{% include 'backends/badges.md' %}

!!! experimental "Introduced in v4.0"

    The Polars backend is experimental and is subject to backwards incompatible changes.

## Install

Install `ibis` and dependencies for the Polars backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[polars]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-polars
    ```

{% endfor %}

## Connect

### `ibis.polars.connect`

```python
con = ibis.polars.connect()
```

<!-- prettier-ignore-start -->
!!! info "`ibis.polars.connect` is a thin wrapper around [`ibis.backends.polars.Backend.do_connect`][ibis.backends.polars.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.polars.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->

## File Support

<!-- prettier-ignore-start -->
::: ibis.backends.polars.Backend.read_csv
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.polars.Backend.read_parquet
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.polars.Backend.read_delta
    options:
      heading_level: 4
      show_docstring_returns: false
<!-- prettier-ignore-end -->
