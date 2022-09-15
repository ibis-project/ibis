import sqlite3

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.sqlite import to_datetime

# Test with formats 1-7 (with T, Z, and offset modifiers) from:
# https://sqlite.org/lang_datefunc.html#time_values
TIMESTAMPS = [
    "2022-01-02",
    "2022-01-02 03:04",
    "2022-01-02 03:04:05",
    "2022-01-02 03:04:05.678",
    "2022-01-02T03:04",
    "2022-01-02T03:04:05",
    "2022-01-02T03:04:05.678",
    None,
]
TIMESTAMPS_TZ = [
    "2022-01-02 03:04Z",
    "2022-01-02 03:04:05Z",
    "2022-01-02 03:04:05.678Z",
    "2022-01-02 03:04+01:00",
    "2022-01-02 03:04:05+01:00",
    "2022-01-02 03:04:05.678+01:00",
    None,
]


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("databases") / "formats.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE timestamps (ts TIMESTAMP)")
    con.execute("CREATE TABLE timestamps_tz (ts TIMESTAMP)")
    with con:
        con.executemany(
            "INSERT INTO timestamps VALUES (?)", [(t,) for t in TIMESTAMPS]
        )
        con.executemany(
            "INSERT INTO timestamps_tz VALUES (?)",
            [(t,) for t in TIMESTAMPS_TZ],
        )
    con.close()
    return path


@pytest.mark.parametrize(
    "table, data",
    [("timestamps", TIMESTAMPS), ("timestamps_tz", TIMESTAMPS_TZ)],
)
def test_timestamps(db, table, data):
    con = ibis.sqlite.connect(db)
    t = con.table(table)
    assert t.ts.type() == dt.timestamp
    res = t.ts.execute()
    sol = pd.Series([to_datetime(s) for s in data]).dt.tz_localize(None)
    assert res.equals(sol)
