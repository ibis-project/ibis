from __future__ import annotations

from time import sleep

import pytest

import ibis
from ibis.backends.pyspark.datatypes import PySparkSchema


@pytest.fixture
def session(streaming_con):
    return streaming_con._session


def test_read_csv(streaming_con, session):
    schema = ibis.schema(
        {
            "createTime": "timestamp",
            "orderId": "int64",
            "payAmount": "float64",
            "payPlatform": "int32",
            "provinceId": "int32",
        }
    )
    t = streaming_con.read_csv(
        "ibis/backends/pyspark/tests/test_streaming/spark-test-data",
        table_name="t",
        schema=PySparkSchema.from_ibis(schema),
        header=True,
    )
    streaming_con.write_to_memory(t, "n")
    sleep(1)
    pd_df = session.sql("SELECT count(*) FROM n").toPandas()
    assert not pd_df.empty
    assert pd_df.iloc[0, 0] == 200
