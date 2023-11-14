---
backend_name: SQLite
backend_url: https://www.sqlite.org/
backend_module: sqlite
backend_param_style: a path to a SQLite database
exclude_backend_api: true
---

{% include 'backends/template.md' %}

## Backend API

<!-- prettier-ignore-start -->
::: ibis.backends.sqlite.Backend
    options:
      heading_level: 3
      inherited_members: true
      members:
        - add_operation
        - attach
        - compile
        - connect
        - create_database
        - create_table
        - create_view
        - database
        - drop_table
        - drop_view
        - execute
        - exists_database
        - exists_table
        - explain
        - insert
        - list_databases
        - list_tables
        - load_data
        - raw_sql
        - schema
        - table
        - verify
<!-- prettier-ignore-end -->
