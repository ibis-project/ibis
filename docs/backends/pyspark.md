# [PySpark](https://spark.apache.org/sql/)

## Install

Install dependencies for Ibis's PySpark dialect:

```sh
pip install 'ibis-framework[pyspark]'
```

or

```sh
conda install -c conda-forge ibis-pyspark
```

## Connect

The PySpark client is accessible through the `ibis.pyspark` namespace.

Use `ibis.pyspark.connect` to create a client.

## API

<!-- prettier-ignore-start -->
::: ibis.backends.pyspark.Backend
    rendering:
      heading_level: 3
    selection:
      filters:
        - "!^_schema_from_csv"

<!-- prettier-ignore-end -->
