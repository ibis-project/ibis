---
backend_name: PySpark
backend_url: https://spark.apache.org/docs/latest/api/python/
backend_module: pyspark
backend_param_style: PySpark things
is_experimental: false
backend_connection_example: |
  >>> import ibis
  >>> ibis.pyspark.connect(...)
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.pyspark.Backend
    rendering:
      heading_level: 3
    selection:
      filters:
        - "!^_schema_from_csv"

<!-- prettier-ignore-end -->
