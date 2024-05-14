from __future__ import annotations

from time import sleep

import pandas as pd
import pytest

from ibis.backends.conftest import TEST_TABLES
from ibis.backends.pyspark.datatypes import PySparkSchema


@pytest.fixture(autouse=True, scope="function")
def stop_active_jobs(session):
    yield
    for sq in session.streams.active:
        sq.stop()
        sq.awaitTermination()


@pytest.fixture
def awards_players_schema():
    return TEST_TABLES["awards_players"]


def test_read_csv_directory(con, session, awards_players_schema):
    t = con.read_csv_directory(
        "ci/ibis-testing-data/directory/csv/awards_players",
        table_name="t",
        schema=PySparkSchema.from_ibis(awards_players_schema),
        header=True,
    )
    con.write_to_memory(t, "n")
    sleep(2)  # wait for results to populate; count(*) returns 0 if executed right away
    pd_df = session.sql("SELECT count(*) FROM n").toPandas()
    assert not pd_df.empty
    assert pd_df.iloc[0, 0] == 6078


def test_read_parquet_directory(con, session):
    t = con.read_parquet_directory(
        "ci/ibis-testing-data/directory/parquet/awards_players", table_name="t"
    )
    con.write_to_memory(t, "n")
    sleep(2)  # wait for results to populate; count(*) returns 0 if executed right away
    pd_df = session.sql("SELECT count(*) FROM n").toPandas()
    assert not pd_df.empty
    assert pd_df.iloc[0, 0] == 6078


def test_to_csv_directory(con, tmp_path):
    t = con.table("awards_players")
    path = tmp_path / "out"
    con.to_csv_directory(
        t.limit(5),
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint", "header": True},
    )
    sleep(2)
    df = pd.concat([pd.read_csv(f) for f in path.glob("*.csv")])
    assert len(df) == 5


def test_to_parquet_directory(con, tmp_path):
    t = con.table("awards_players")
    path = tmp_path / "out"
    con.to_parquet_directory(
        t.limit(5),
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint"},
    )
    sleep(2)
    df = pd.concat([pd.read_parquet(f) for f in path.glob("*.parquet")])
    assert len(df) == 5
