from __future__ import annotations

import pandas.testing as tm
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from ibis.backends.pyspark.datatypes import PySparkSchema  # noqa: E402


@pytest.fixture
def awards_players(con):
    return con.table("awards_players")


def test_table_to_csv(tmp_path, awards_players):
    outcsv = tmp_path / "out.csv"

    # avoid pandas NaNonense
    columns = ["playerID", "awardID", "yearID", "lgID"]
    awards_players = awards_players.select(columns)

    awards_players.to_csv(outcsv)

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(
        str(outcsv), schema=PySparkSchema.from_ibis(awards_players.schema())
    ).toPandas()

    tm.assert_frame_equal(
        awards_players.to_pandas().sort_values(by=columns).reset_index(drop=True),
        df.sort_values(by=columns).reset_index(drop=True),
    )


@pytest.mark.parametrize("delimiter", [";", "\t"], ids=["semicolon", "tab"])
def test_table_to_csv_writer_kwargs(delimiter, tmp_path, awards_players):
    outcsv = tmp_path / "out.csv"
    # avoid pandas NaNonense
    awards_players = awards_players.select("playerID", "awardID", "yearID", "lgID")

    awards_players.to_csv(outcsv, sep=delimiter)
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(str(outcsv), sep=delimiter).limit(1).toPandas()
    assert len(df) == 1
