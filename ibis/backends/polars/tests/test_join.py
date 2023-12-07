from __future__ import annotations

import pandas as pd
import pandas.testing as tm

import ibis


def test_memtable_join(con):
    t1 = ibis.memtable({"x": [1, 2, 3], "y": [4, 5, 6], "z": ["a", "b", "c"]})
    t2 = ibis.memtable({"x": [3, 2, 1], "y": [7, 8, 9], "z": ["d", "e", "f"]})

    expected = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": ["a", "b", "c"],
            "y_right": [9, 8, 7],
            "z_right": ["f", "e", "d"],
        }
    )

    result = con.execute(t1.join(t2, "x"))
    tm.assert_frame_equal(result, expected)
