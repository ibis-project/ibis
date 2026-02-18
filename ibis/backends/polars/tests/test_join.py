from __future__ import annotations

import pandas as pd
import pandas.testing as tm

import ibis


def test_memtable_join(con):
    t1 = ibis.memtable({"x": [1, 2, 3], "y": [4, 5, 6], "z": ["a", "b", "c"]})
    t2 = ibis.memtable({"x": [3, 2, 1], "y": [7, 8, 9], "z": ["d", "e", "f"]})
    expr = t1.join(t2, "x")

    result = con.execute(expr)
    expected = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": ["a", "b", "c"],
            "y_right": [9, 8, 7],
            "z_right": ["f", "e", "d"],
        }
    )

    left = result.sort_values("x").reset_index(drop=True)
    right = expected.sort_values("x").reset_index(drop=True)
    tm.assert_frame_equal(left, right)


def test_memtable_join_left(con):
    t1 = ibis.memtable({"x": [1, 2, 3], "y": [4, 5, 6], "z": ["a", "b", "c"]})
    t2 = ibis.memtable({"x": [3, 2, 1], "y": [7, 8, 9], "z": ["d", "e", "f"]})
    expr = t1.join(t2, "x", how="left")

    result = con.execute(expr)
    expected = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": ["a", "b", "c"],
            "x_right": [1, 2, 3],
            "y_right": [9, 8, 7],
            "z_right": ["f", "e", "d"],
        }
    )

    left = result.sort_values("x").reset_index(drop=True)
    right = expected.sort_values("x").reset_index(drop=True)
    tm.assert_frame_equal(left, right)


def test_memtable_join_cross(con):
    t1 = ibis.memtable({"x": [1, 2, 3], "y": [4, 5, 6], "z": ["a", "b", "c"]})
    t2 = ibis.memtable({"x": [3, 2, 1], "y": [7, 8, 9], "z": ["d", "e", "f"]})
    expr = t1.join(t2, how="cross")

    result = con.execute(expr)
    expected = pd.DataFrame(
        {
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "y": [4, 4, 4, 5, 5, 5, 6, 6, 6],
            "z": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "x_right": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y_right": [9, 8, 7, 9, 8, 7, 9, 8, 7],
            "z_right": ["f", "e", "d", "f", "e", "d", "f", "e", "d"],
        }
    )

    left = result.sort_values(["x", "x_right"]).reset_index(drop=True)
    right = expected.sort_values(["x", "x_right"]).reset_index(drop=True)
    tm.assert_frame_equal(left, right)
