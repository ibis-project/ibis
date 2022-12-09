---
hide:
  - toc
---

# Operation Support Matrix

Backends are shown in descending order of the number of supported operations.

!!! tip "The Snowflake backend coverage is an overestimate"

    The Snowflake backend translation functions are reused from the PostgreSQL backend
    and some operations that claim coverage may not work.

    The Snowflake backend is a good place to start contributing!

## Core Operations

{{ read_csv("docs/backends/core_support_matrix.csv") }}

## Geospatial Operations

{{ read_csv("docs/backends/geospatial_support_matrix.csv") }}
