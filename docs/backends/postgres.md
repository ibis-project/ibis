---
backend_name: PostgreSQL
backend_url: https://www.postgresql.org/
backend_module: postgres
backend_param_style: a SQLAlchemy-style URI
is_experimental: false
backend_connection_example: |
  >>> con = ibis.postgres.connect(url='postgresql://postgres:postgres@postgres:5432/ibis_testing')
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.postgres.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
