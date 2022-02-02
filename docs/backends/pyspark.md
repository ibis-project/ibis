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

!!! note "Check your PySpark version"

    When using the PySpark backend with PySpark 2.3.x, 2.4.x and pyarrow >=
    0.15.0, you need to set `ARROW_PRE_0_15_IPC_FORMAT=1`. See
    [here](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
    for details

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
