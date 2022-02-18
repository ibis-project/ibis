---
backend_name: MySQL
backend_url: https://www.mysql.com/
backend_module: mysql
backend_param_style: a SQLAlchemy-style URI
is_experimental: false
backend_connection_example: |
  >>> con = ibis.mysql.connect(url='mysql+pymysql://ibis:ibis@mysql/ibis_testing')
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.mysql.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
