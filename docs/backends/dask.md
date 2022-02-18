---
backend_name: Dask
backend_url: https://dask.org
backend_module: dask
backend_param_style: a dictionary of paths
is_experimental: true
backend_connection_example: |
  >>> import ibis
  >>> data_sources = {"t": "path/to/file.parquet", "s": "path/to/file.csv"}
  >>> client = ibis.dask.connect(data_sources)
  >>> t = client.table("t")
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.dask.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
