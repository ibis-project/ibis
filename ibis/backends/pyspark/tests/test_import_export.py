from __future__ import annotations

from operator import methodcaller
from time import sleep
from unittest import mock

import pandas as pd
import pytest

from ibis.backends.conftest import TEST_TABLES
from ibis.backends.pyspark import Backend
from ibis.backends.pyspark.datatypes import PySparkSchema


@pytest.fixture(scope="session", autouse=True)
def default_session_fixture():
    with mock.patch.object(Backend, "write_to_memory", write_to_memory, create=True):
        yield


def write_to_memory(self, expr, table_name):
    if self.mode == "batch":
        raise NotImplementedError
    df = self._session.sql(expr.compile())
    df.writeStream.format("memory").queryName(table_name).start()


@pytest.fixture(autouse=True, scope="function")
def stop_active_jobs(con_streaming):
    yield
    for sq in con_streaming._session.streams.active:
        sq.stop()
        sq.awaitTermination()


@pytest.fixture
def awards_players_schema():
    return TEST_TABLES["awards_players"]


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
