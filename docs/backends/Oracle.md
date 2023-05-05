---
backend_name: Oracle
backend_url: https://docs.oracle.com/en/database/oracle/oracle-database/index.html
backend_module: oracle
backend_param_style: a SQLAlchemy connection string
backend_connection_example: ibis.connect("oracle://user:pass@host:port/service_name")
is_experimental: true
version_added: "6.0"
---

{% include 'backends/template.md' %}
