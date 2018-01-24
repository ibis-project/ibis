import pytest

import ibis.tests.util as tu


@pytest.mark.parametrize('column', [
    'string_col',
    'double_col',
    'date_string_col',
    pytest.mark.skip(
        pytest.mark.xfail('timestamp_col',
                          raises=AssertionError,
                          reason='NYT'),
        reason='hangs'
    )
])
@tu.skipif_unsupported
def test_distinct_column(backend, alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)
