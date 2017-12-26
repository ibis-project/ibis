import pytest

import ibis.tests.util as tu


@pytest.mark.parametrize(
    'column',
    ['string_col', 'double_col', 'date_string_col', 'timestamp_col']
)
@tu.skip_if_invalid_operation
def test_distinct_column(alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)
