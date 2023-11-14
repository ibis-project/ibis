---
backend_name: DuckDB
backend_url: https://duckdb.org/
backend_module: duckdb
backend_param_style: a path to a DuckDB database
backend_connection_example: ibis.duckdb.connect("path/to/my.duckdb")
version_added: "3.0"
intro: |
  !!! danger "`duckdb` >= 0.5.0 requires `duckdb-engine` >= 0.6.2"

      If you encounter problems when using `duckdb` >= **0.5.0** you may need to
      upgrade `duckdb-engine` to at least version **0.6.2**.

      See [this issue](https://github.com/ibis-project/ibis/issues/4503) for
      more details.
---

{% include 'backends/template.md' %}
