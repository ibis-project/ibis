---
backend_name: Snowflake
backend_url: https://snowflake.com/
backend_module: snowflake
backend_param_style: a SQLAlchemy connection string
backend_connection_example: ibis.connect("snowflake://user:pass@locator/database/schema")
is_experimental: true
version_added: "4.0"
---

{% include 'backends/template.md' %}
