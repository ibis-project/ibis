# from __future__ import annotations

# import numpy as np
# import pandas as pd
# import pytest

# from ibis.backends.pandas.newwindows import RowsWindowIterator


# @pytest.fixture
# def df():
#     return pd.DataFrame({"col": range(10)})


# @pytest.mark.parametrize(
#     ("start_offset", "end_offset", "expected_windows"),
#     [
#         (
#             0,
#             2,
#             [
#                 [0, 1, 2],
#                 [1, 2, 3],
#                 [2, 3, 4],
#                 [3, 4, 5],
#                 [4, 5, 6],
#                 [5, 6, 7],
#                 [6, 7, 8],
#                 [7, 8, 9],
#             ],
#         ),
#         (
#             -1,
#             2,
#             [
#                 [np.nan, 0, 1, 2],
#                 [0, 1, 2, 3],
#                 [1, 2, 3, 4],
#                 [2, 3, 4, 5],
#                 [3, 4, 5, 6],
#                 [4, 5, 6, 7],
#                 [5, 6, 7, 8],
#                 [6, 7, 8, 9],
#             ],
#         ),
#         (
#             1,
#             2,
#             [
#                 [1, 2],
#                 [2, 3],
#                 [3, 4],
#                 [4, 5],
#                 [5, 6],
#                 [6, 7],
#                 [7, 8],
#                 [8, 9],
#                 [9, np.nan],
#             ],
#         ),
#     ],
# )
# def test_rows_window_iterator(start_offset, end_offset, expected_windows, df):
#     it = RowsWindowIterator(
#         df, start_offset, end_offset, prepend_nan=True, append_nan=True
#     )

#     for i, (window, expected) in enumerate(zip(it, expected_windows)):
#         e = np.array(expected)
#         f = e[:, np.newaxis]
#         assert np.array_equal(window, f, equal_nan=True)
