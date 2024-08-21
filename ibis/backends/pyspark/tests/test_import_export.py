from __future__ import annotations

from operator import methodcaller
from time import sleep

import pandas as pd
import pytest

from ibis.backends.pyspark.datatypes import PySparkSchema


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("read_delta", path="test.delta"),
        methodcaller("read_csv", source_list="test.csv"),
        methodcaller("read_parquet", path="test.parquet"),
        methodcaller("read_json", source_list="test.json"),
    ],
)
def test_streaming_import_not_implemented(con_streaming, method):
    with pytest.raises(NotImplementedError):
        method(con_streaming)


def test_read_csv_dir(con_streaming, awards_players_schema):
    t = con_streaming.read_csv_dir(
        "ci/ibis-testing-data/directory/csv/awards_players",
        table_name="t",
        schema=PySparkSchema.from_ibis(awards_players_schema),
        header=True,
    )
    con_streaming.write_to_memory(t, "n")
    sleep(2)  # wait for results to populate; count(*) returns 0 if executed right away
    pd_df = con_streaming._session.sql("SELECT count(*) FROM n").toPandas()
    assert not pd_df.empty
    assert pd_df.iloc[0, 0] == 6078


def test_read_parquet_dir(con_streaming):
    t = con_streaming.read_parquet_dir(
        "ci/ibis-testing-data/directory/parquet/awards_players", table_name="t"
    )
    con_streaming.write_to_memory(t, "n")
    sleep(2)  # wait for results to populate; count(*) returns 0 if executed right away
    pd_df = con_streaming._session.sql("SELECT count(*) FROM n").toPandas()
    assert not pd_df.empty
    assert pd_df.iloc[0, 0] == 6078


def test_to_csv_dir(con_streaming, tmp_path):
    t = con_streaming.table("awards_players")
    path = tmp_path / "out"
    con_streaming.to_csv_dir(
        t.limit(5),
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint", "header": True},
    )
    sleep(2)
    df = pd.concat([pd.read_csv(f) for f in path.glob("*.csv")])
    assert len(df) == 5


def test_to_parquet_dir(con_streaming, tmp_path):
    t = con_streaming.table("awards_players")
    path = tmp_path / "out"
    con_streaming.to_parquet_dir(
        t.limit(5),
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint"},
    )
    sleep(2)
    df = pd.concat([pd.read_parquet(f) for f in path.glob("*.parquet")])
    assert len(df) == 5
