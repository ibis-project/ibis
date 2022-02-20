---
backend_name: Dask
backend_url: https://dask.org
backend_module: dask
backend_param_style: a dictionary of paths
is_experimental: true
backend_connection_example: 'ibis.dask.connect({"t": "path/to/file.parquet", "s": "path/to/file.csv"})'
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.dask.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
