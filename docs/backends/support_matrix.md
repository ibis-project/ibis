---
hide:
  - toc
---

# Operation Support Matrix

Backends are shown in descending order of the number of supported operations.

!!! tip "Backends with low coverage are good places to start contributing!"

    Each backend implements operations differently, but this is usually very similar
    to other backends.
    If you want to start contributing to ibis, it's a good idea to start by adding missing operations
    to backends that have low operation coverage.

## Core Operations

{{ read_csv("docs/backends/core_support_matrix.csv") }}

## Geospatial Operations

{{ read_csv("docs/backends/geospatial_support_matrix.csv") }}

## Raw Data

You can also download data from the above tables in [CSV format](./raw_support_matrix.csv).
