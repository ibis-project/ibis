---
backend_name: Datafusion
backend_url: https://arrow.apache.org/datafusion/
backend_module: datafusion
backend_param_style: a dictionary of paths
is_experimental: true
version_added: "2.1"
backend_connection_example: 'ibis.datafusion.connect({"t": "path/to/file.parquet", "s": "path/to/file.csv"})'
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.datafusion.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
