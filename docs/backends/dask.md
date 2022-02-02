# [Dask](https://dask.org/)

!!! experimental

    The Dask backend is experimental.

## Install

Install ibis along with its dependencies for the datafusion backend:

```sh
pip install 'ibis-framework[dask]'
```

```sh
conda install -c conda-forge ibis-dask
```

## Connect

Create a client by passing a dictionary that maps table names to paths to
`ibis.dask.connect`:

```python
>>> import ibis
>>> data_sources = {"t": "path/to/file.parquet", "s": "path/to/file.csv"}
>>> client = ibis.dask.connect(data_sources)
>>> t = client.table("t")
```

## API

The Dask client is accessible through the `ibis.dask` namespace.

Use `ibis.dask.connect` to create a Dask client.

<!-- prettier-ignore-start -->
::: ibis.backends.dask.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
