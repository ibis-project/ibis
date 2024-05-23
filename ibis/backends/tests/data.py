from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

array_types = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3], None],
        ),
        (
            [4, 5],
            ["d", "e"],
            [4.0, 5.0],
            "a",
            2.0,
            [],
        ),
        (
            [6, None],
            ["f", None],
            [6.0, np.nan],
            "a",
            3.0,
            [None, [], None],
        ),
        (
            [None, 1, None],
            [None, "a", None],
            [],
            "b",
            4.0,
            [[1], [2], [], [3, 4, 5]],
        ),
        (
            [2, None, 3],
            ["b", None, "c"],
            np.nan,
            "b",
            5.0,
            None,
        ),
        (
            [4, None, None, 5],
            ["d", None, None, "e"],
            [4.0, np.nan, np.nan, 5.0],
            "c",
            6.0,
            [[1, 2, 3]],
        ),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)

non_null_array_types = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3]],
        ),
        ([4, 5], ["d", "e"], [4.0, 5.0], "a", 2.0, []),
        ([6], ["f"], [6.0], "a", 3.0, [[]]),
        ([1], ["a"], [], "b", 4.0, [[1], [2], [], [3, 4, 5]]),
        ([2, 3], ["b", "c"], np.nan, "b", 5.0, None),
        ([4, 5], ["d", "e"], [4.0, 5.0], "c", 6.0, [[1, 2, 3]]),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)

json_types = pd.DataFrame(
    {
        "js": [
            '{"a": [1,2,3,4], "b": 1}',
            '{"a":null,"b":2}',
            '{"a":"foo", "c":null}',
            "null",
            "[42,47,55]",
            "[]",
            '"a"',
            '""',
            '"b"',
            None,
            "true",
            "false",
            "42",
            "37.37",
        ]
    }
)

struct_types = pd.DataFrame(
    {
        "abc": [
            {"a": 1.0, "b": "banana", "c": 2},
            {"a": 2.0, "b": "apple", "c": 3},
            {"a": 3.0, "b": "orange", "c": 4},
            {"a": pd.NA, "b": "banana", "c": 2},
            {"a": 2.0, "b": pd.NA, "c": 3},
            pd.NA,
            {"a": 3.0, "b": "orange", "c": pd.NA},
        ]
    }
)

win = pd.DataFrame(
    {
        "g": ["a", "a", "a", "a", "a"],
        "x": [0, 1, 2, 3, 4],
        "y": [3, 2, 0, 1, 1],
    }
)

topk = pa.Table.from_pydict({"x": [1, 1, None]})

# a time series dataset for streaming use cases
payment_msg = pd.DataFrame(
    {
        "createTime": {
            0: pd.Timestamp("2024-05-14 22:45:17.481"),
            1: pd.Timestamp("2024-05-14 22:45:17.481"),
            2: pd.Timestamp("2024-05-14 22:45:17.481"),
            3: pd.Timestamp("2024-05-14 22:45:17.481"),
            4: pd.Timestamp("2024-05-14 22:45:17.481"),
        },
        "orderId": {
            0: 1715726718,
            1: 1715726719,
            2: 1715726720,
            3: 1715726721,
            4: 1715726722,
        },
        "payAmount": {
            0: 19804.010084996993,
            1: 95161.77724974867,
            2: 80639.50276056337,
            3: 38627.754005951545,
            4: 87939.00723568315,
        },
        "payPlatform": {0: 1, 1: 0, 2: 0, 3: 0, 4: 0},
        "provinceId": {0: 2, 1: 6, 2: 3, 3: 2, 4: 1},
    }
)
