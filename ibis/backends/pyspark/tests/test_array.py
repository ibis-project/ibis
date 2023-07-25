from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis

pytest.importorskip("pyspark")


def test_array_length(con):
    table = con.table("array_table")

    result = table.mutate(length=table.array_int.length()).compile()

    expected = table.compile().toPandas()
    expected["length"] = expected["array_int"].map(lambda a: len(a)).astype("int32")
    tm.assert_frame_equal(result.toPandas(), expected)


def test_array_length_scalar(con):
    raw_value = [1, 2, 3]
    value = ibis.literal(raw_value)
    expr = value.length()
    result = con.execute(expr)
    expected = len(raw_value)
    assert result == expected


@pytest.mark.parametrize(
    ["start", "stop"],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (None, -3),
        (-3, -1),
    ],
)
def test_array_slice(con, start, stop):
    table = con.table("array_table")

    result = table.mutate(sliced=table.array_int[start:stop]).compile()

    expected = table.compile().toPandas()
    expected["sliced"] = expected["array_int"].map(lambda a: a[start:stop])
    tm.assert_frame_equal(result.toPandas(), expected)


@pytest.mark.parametrize(
    ["start", "stop"],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (None, -3),
        (-3, -1),
    ],
)
def test_array_slice_scalar(con, start, stop):
    raw_value = [-11, 42, 10]
    value = ibis.literal(raw_value)
    expr = value[start:stop]
    result = con.execute(expr)
    expected = raw_value[start:stop]
    assert result == expected


@pytest.mark.parametrize("index", [1, 3, 4, 11, -11])
def test_array_index(con, index):
    table = con.table("array_table")
    expr = table[table.array_int[index].name("indexed")]
    result = expr.execute()

    df = table.compile().toPandas()
    expected = pd.DataFrame(
        {
            "indexed": df.array_int.apply(
                lambda x: x[index] if -len(x) <= index < len(x) else np.nan
            )
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index", [1, 3, 4, 11])
def test_array_index_scalar(con, index):
    raw_value = [-10, 1, 2, 42]
    value = ibis.literal(raw_value)
    expr = value[index]
    result = con.execute(expr)
    expected = raw_value[index] if index < len(raw_value) else np.nan
    assert result == expected or (np.isnan(result) and np.isnan(expected))


@pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat(con, op):
    table = con.table("array_table")
    x = table.array_int.cast("array<string>")
    y = table.array_str
    expr = op(x, y).name("array_result")
    result = expr.execute()

    df = table.compile().toPandas()
    expected = op(df.array_int.apply(lambda x: list(map(str, x))), df.array_str).rename(
        "array_result"
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat_scalar(con, op):
    raw_left = [1, 2, 3]
    raw_right = [3, 4]
    left = ibis.literal(raw_left)
    right = ibis.literal(raw_right)
    expr = op(left, right)
    result = con.execute(expr)
    assert result == op(raw_left, raw_right)


@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize("mul", [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat(con, n, mul):
    table = con.table("array_table")

    expr = table.select(mul(table.array_int, n).name("repeated"))
    result = expr.execute()

    df = table.compile().toPandas()
    expected = pd.DataFrame({"repeated": df.array_int * n})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize("mul", [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat_scalar(con, n, mul):
    raw_array = [1, 2]
    array = ibis.literal(raw_array)
    expr = mul(array, n)
    result = con.execute(expr)
    expected = mul(raw_array, n)
    assert result == expected


def test_array_collect(con):
    table = con.table("array_table")
    expr = table.group_by(table.key).aggregate(collected=table.array_int.collect())
    result = expr.execute().sort_values("key").reset_index(drop=True)

    df = table.compile().toPandas()
    expected = (
        df.groupby("key")
        .array_int.apply(list)
        .reset_index()
        .rename(columns={"array_int": "collected"})
    )
    tm.assert_frame_equal(result, expected)


def test_array_filter(con):
    table = con.table("array_table")
    expr = table.select(
        table.array_int.filter(lambda item: item != 3).name("array_int")
    )
    result = expr.execute()
    df = table.compile().toPandas()
    df["array_int"] = df["array_int"].apply(
        lambda ar: [item for item in ar if item != 3]
    )
    expected = df[["array_int"]]
    tm.assert_frame_equal(result, expected)
