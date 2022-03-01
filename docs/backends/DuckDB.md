---
backend_name: DuckDB
backend_url: https://duckdb.org/
backend_module: duckdb
backend_param_style: a path to a DuckDB database
backend_connection_example: ibis.duckdb.connect("path/to/my.duckdb")
development_only: true
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.duckdb.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
