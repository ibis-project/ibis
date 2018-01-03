import pytest


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
def test_distinct_column(backend, alltypes, df, column):
    expr = alltypes[column].distinct()
    expected = df[column].unique()

    with backend.skip_unsupported():
        result = expr.execute()

    assert set(result) == set(expected)
