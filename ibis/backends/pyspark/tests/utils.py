from __future__ import annotations

from typing import TYPE_CHECKING

import ibis

if TYPE_CHECKING:
    import pandas as pd


def str_to_struct(column: str, type_: str):
    from pyspark.sql.types import StructField

    from ibis.backends.pyspark.datatypes import PySparkType

    ibis_type = ibis.dtype(type_)
    pyspark_type = PySparkType.from_ibis(ibis_type)

    return StructField(column, pyspark_type)


def spark_schema_from_df(df: pd.DataFrame):
    from pyspark.sql.types import StructType

    return StructType(
        [
            str_to_struct(column, type_)
            for column, type_ in zip(list(df.columns), list(df.dtypes))
        ]
    )
