import tempfile
from pathlib import Path

import pandas as pd
import pytest

import ibis


def test_read_csv(data_directory):
    t = ibis.read_csv(data_directory / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_parquet(data_directory):
    t = ibis.read_parquet(data_directory / "functional_alltypes.parquet")
    assert t.count().execute()


def test_temp_directory(tmp_path):
    query = "SELECT value FROM duckdb_settings() WHERE name = 'temp_directory'"

    # 1. in-memory + no temp_directory specified
    con = ibis.duckdb.connect()
    [(value,)] = con.con.execute(query).fetchall()
    assert value  # we don't care what the specific value is

    temp_directory = Path(tempfile.gettempdir()) / "duckdb"

    # 2. in-memory + temp_directory specified
    con = ibis.duckdb.connect(temp_directory=temp_directory)
    [(value,)] = con.con.execute(query).fetchall()
    assert value == str(temp_directory)

    # 3. on-disk + no temp_directory specified
    # untested, duckdb sets the temp_directory to something implementation
    # defined

    # 4. on-disk + temp_directory specified
    con = ibis.duckdb.connect(tmp_path / "test2.ddb", temp_directory=temp_directory)
    [(value,)] = con.con.execute(query).fetchall()
    assert value == str(temp_directory)


# Skipping this to avoid CI shenanigans for the moment but it passes
# as of 2023-01-05
@pytest.mark.skipif(True, reason="avoiding CI shenanigans")
def test_read_postgres():
    con = ibis.postgres.connect(
        host="localhost", port=5432, user="postgres", password="postgres"
    )
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1.0], "y": ["a", "b", "c", "a"]})
    s = ibis.schema(dict(x="float64", y="str"))

    con.create_table("duckdb_test", df, s)

    con = ibis.duckdb.connect()
    uri = "postgres://postgres:postgres@localhost:5432"
    table = con.read_postgres(uri, "duckdb_test")
    assert table.count().execute()
