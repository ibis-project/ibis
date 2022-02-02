# [ClickHouse](https://clickhouse.yandex/)

## Install

Install dependencies for Ibis's ClickHouse dialect (minimal supported version is `0.1.3`):

```sh
pip install 'ibis-framework[clickhouse]'
```

```sh
conda install -c conda-forge ibis-clickhouse
```

## Connect

Create a client by passing in database connection parameters such as `host`,
`port`, `database`, and `user` to :func:`ibis.clickhouse.connect`:

```python
con = ibis.clickhouse.connect(host='clickhouse', port=9000)
```

## API

The ClickHouse client is accessible through the `ibis.clickhouse` namespace.

Use `ibis.clickhouse.connect` to create a client.

<!-- prettier-ignore-start -->
::: ibis.backends.clickhouse.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
