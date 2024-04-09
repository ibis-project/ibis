from __future__ import annotations

import pandas.testing as tm
import pytest

pyspark = pytest.importorskip("pyspark")


@pytest.mark.parametrize(
    "table_name",
    [
        "null_table",
        "null_table_streaming",
    ],
)
def test_isnull(con, table_name):
    table = con.table(table_name)
    table_pandas = table.execute()

    for col, _ in table_pandas.items():
        result = table[table[col].isnull()].execute().reset_index(drop=True)
        expected = table_pandas[table_pandas[col].isnull()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "table_name",
    [
        "null_table",
        "null_table_streaming",
    ],
)
def test_notnull(con, table_name):
    table = con.table(table_name)
    table_pandas = table.execute()

    for col, _ in table_pandas.items():
        result = table[table[col].notnull()].execute().reset_index(drop=True)
        expected = table_pandas[table_pandas[col].notnull()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
