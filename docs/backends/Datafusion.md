---
backend_name: Datafusion
backend_url: https://arrow.apache.org/datafusion/
backend_module: datafusion
version_added: "2.1"
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
imports: ["CSV", "Parquet"]
---

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the Apache Datafusion backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[datafusion]`
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-datafusion
    ```

{% endfor %}

## Connect

### `ibis.datafusion.connect`

```python
con = ibis.datafusion.connect()
```

```python
con = ibis.datafusion.connect(
    config={"table1": "path/to/file.parquet", "table2": "path/to/file.csv"}
)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.datafusion.connect` is a thin wrapper around [`ibis.backends.datafusion.Backend.do_connect`][ibis.backends.datafusion.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.datafusion.Backend.do_connect
    options:
      heading_level: 4
      show_docstring_examples: false
<!-- prettier-ignore-end -->

## File Support

<!-- prettier-ignore-start -->
::: ibis.backends.datafusion.Backend.read_csv
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.datafusion.Backend.read_parquet
    options:
      heading_level: 4
      show_docstring_returns: false
<!-- prettier-ignore-end -->
