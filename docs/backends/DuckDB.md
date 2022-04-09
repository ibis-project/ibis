---
backend_name: DuckDB
backend_url: https://duckdb.org/
backend_module: duckdb
backend_param_style: a path to a DuckDB database
backend_connection_example: ibis.duckdb.connect("path/to/my.duckdb")
development_only: false
---

{% include 'backends/template.md' %}
