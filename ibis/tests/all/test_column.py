"""Tests for operations that work across columns of all (or most) types.
"""

import pytest

import ibis.tests.util as tu


@pytest.mark.parametrize(
    'column',
    [
        'string_col',
        'double_col',
        'date_string_col',
        pytest.mark.xfail('timestamp_col', raises=AssertionError, reason='NYT')
    ]
)
@tu.skip_if_invalid_operation
@pytest.mark.backend
def test_distinct_column(backend, backend_alltypes, backend_df, column):
    expr = backend_alltypes[column].distinct()
    result = expr.execute()
    expected = backend_df[column].unique()
    assert frozenset(result) == frozenset(expected)
