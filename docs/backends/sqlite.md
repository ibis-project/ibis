---
backend_name: SQLite
backend_url: https://www.sqlite.org/
backend_module: sqlite
backend_param_style: a path to a SQLite database
is_experimental: true
backend_connection_example: |
  >>> import ibis
  >>> ibis.sqlite.connect('path/to/my/sqlite.db')
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.sqlite.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
