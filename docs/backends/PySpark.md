---
backend_name: PySpark
backend_url: https://spark.apache.org/docs/latest/api/python/
backend_module: pyspark
backend_param_style: PySpark things
backend_connection_example: |
  session = pyspark.sql.SparkSession.builder.getOrCreate()
  ibis.pyspark.connect(session)
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
