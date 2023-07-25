from __future__ import annotations

import pandas.testing as tm
import pytest

pytest.importorskip("pyspark")


def test_isnull(con):
    table = con.table("null_table")
    table_pandas = table.compile().toPandas()

    for col, _ in table_pandas.items():
        result = table[table[col].isnull()].compile().toPandas().reset_index(drop=True)
        expected = table_pandas[table_pandas[col].isnull()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


def test_notnull(con):
    table = con.table("null_table")
    table_pandas = table.compile().toPandas()

    for col, _ in table_pandas.items():
        result = table[table[col].notnull()].compile().toPandas().reset_index(drop=True)
        expected = table_pandas[table_pandas[col].notnull()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
