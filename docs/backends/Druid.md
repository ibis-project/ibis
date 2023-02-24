---
backend_name: Druid
backend_url: https://druid.apache.org/
backend_module: druid
backend_param_style: a SQLAlchemy connection string
backend_connection_example: ibis.connect("druid://localhost:8082/druid/v2/sql")
is_experimental: true
version_added: "5.0"
---

{% include 'backends/template.md' %}
