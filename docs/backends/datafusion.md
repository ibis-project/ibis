# [Datafusion](https://arrow.apache.org/datafusion/)

!!! experimental "New in v2.1"

    The Datafusion backend is experimental.

## Install

Install ibis along with its dependencies for the datafusion backend:

```sh
pip install 'ibis-framework[datafusion]'
```

```sh
conda install -c conda-forge ibis-datafusion
```

## Connect

Create a client by passing a dictionary that maps table names to paths to
`ibis.datafusion.connect`:

```python
>>> import ibis
>>> data_sources = {"t": "path/to/file.parquet", "s": "path/to/file.csv"}
>>> client = ibis.datafusion.connect(data_sources)
>>> t = client.table("t")
```

## API

The Datafusion client is accessible through the `ibis.datafusion` namespace.

Use `ibis.datafusion.connect` to create a Datafusion client.

<!-- prettier-ignore-start -->
::: ibis.backends.datafusion.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
