---
backend_name: PySpark
backend_url: https://spark.apache.org/docs/latest/api/python/
backend_module: pyspark
backend_param_style: PySpark things
exports: ["PyArrow", "Parquet", "CSV", "Pandas"]
imports: ["CSV", "Parquet"]
---

{% include 'backends/badges.md' %}

## Install

Install `ibis` and dependencies for the PySpark backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[pyspark]'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-pyspark
    ```

{% endfor %}

## Connect

### `ibis.pyspark.connect`

```python
con = ibis.pyspark.connect(session=session)
```

<!-- prettier-ignore-start -->
!!! info "`ibis.pyspark.connect` is a thin wrapper around [`ibis.backends.pyspark.Backend.do_connect`][ibis.backends.pyspark.Backend.do_connect]."
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
!!! info "The `pyspark` backend does not create `SparkSession` objects, you must create a `SparkSession` and pass that to `ibis.pyspark.connect`."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.pyspark.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->

## File Support

<!-- prettier-ignore-start -->
::: ibis.backends.pyspark.Backend.read_csv
    options:
      heading_level: 4
      show_docstring_returns: false
::: ibis.backends.pyspark.Backend.read_parquet
    options:
      heading_level: 4
      show_docstring_returns: false
<!-- prettier-ignore-end -->
