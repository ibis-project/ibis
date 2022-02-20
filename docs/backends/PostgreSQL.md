---
backend_name: PostgreSQL
backend_url: https://www.postgresql.org/
backend_module: postgres
backend_param_style: a SQLAlchemy-style URI
backend_connection_example: ibis.postgres.connect(url='postgresql://user:pass@host:port/db')
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.postgres.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
