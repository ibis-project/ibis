from ibis.spark.client import _SPARK_DTYPE_TO_IBIS_DTYPE


def test_spark_dtype_to_ibis_dtype():
    assert len(_SPARK_DTYPE_TO_IBIS_DTYPE.keys()) == \
        len(set(_SPARK_DTYPE_TO_IBIS_DTYPE.values()))
