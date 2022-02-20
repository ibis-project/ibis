---
backend_name: ClickHouse
backend_url: https://clickhouse.yandex/
backend_module: clickhouse
backend_param_style: connection parameters
backend_connection_example: ibis.clickhouse.connect(host="localhost", port=9000)
---

{% include 'backends/template.md' %}

## API

<!-- prettier-ignore-start -->
::: ibis.backends.clickhouse.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
