from __future__ import annotations

import pytest

from ibis.backends.pandas.helpers import RowsFrame

lst = list(range(10))


@pytest.mark.parametrize(
    ("ix", "start", "end", "expected"),
    [
        (0, None, None, lst),
        (0, 0, None, lst),
        (0, None, 0, [0]),
        (0, 0, 0, [0]),
        (0, 0, 1, [0, 1]),
        (0, 1, 1, [1]),
        (0, 1, 2, [1, 2]),
        (0, 1, None, lst[1:]),
        (0, None, 1, [0, 1]),
        (0, -1, None, lst),
        (0, None, -1, []),
        (0, -1, -1, []),
        (0, -2, -1, []),
        (0, -2, None, lst),
        (0, None, -2, []),
        (0, -1, 1, [0, 1]),
        (0, 1, -1, []),
        (0, -1, 2, [0, 1, 2]),
        (1, None, None, lst),
        (1, 0, None, lst[1:]),
        (1, None, 0, [0, 1]),
        (1, 0, 0, [1]),
        (1, 0, 1, [1, 2]),
        (1, 1, 1, [2]),
        (1, 1, 2, [2, 3]),
        (1, 1, None, lst[2:]),
        (1, None, 1, [0, 1, 2]),
        (1, -1, None, lst),
        (1, None, -1, [0]),
        (1, -1, -1, [0]),
        (1, -2, -1, [0]),
        (1, -2, None, lst),
        (1, None, -2, []),
        (1, -1, 1, [0, 1, 2]),
        (1, 1, -1, []),
        (1, -1, 2, [0, 1, 2, 3]),
        (2, None, None, lst),
        (2, 0, None, lst[2:]),
        (2, None, 0, [0, 1, 2]),
        (2, 0, 0, [2]),
        (2, 0, 1, [2, 3]),
        (2, 1, 1, [3]),
        (2, 1, 2, [3, 4]),
        (2, 1, None, lst[3:]),
        (2, None, 1, [0, 1, 2, 3]),
        (2, -1, None, lst[1:]),
        (2, None, -1, [0, 1]),
        (2, -1, -1, [1]),
        (2, -2, -1, [0, 1]),
        (2, -2, None, lst),
        (2, None, -2, [0]),
        (2, -1, 1, [1, 2, 3]),
        (2, 1, -1, []),
        (2, -1, 2, [1, 2, 3, 4]),
        (3, None, None, lst),
    ],
)
def test_rows_frame_adjustment(ix, start, end, expected):
    start_index, end_index = RowsFrame.adjust(len(lst), ix, start, end)
    assert lst[start_index:end_index] == expected
