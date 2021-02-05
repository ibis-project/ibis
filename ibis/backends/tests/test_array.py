import numpy as np
import pytest

import ibis


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_array_column(backend, alltypes, df):
    expr = ibis.array_column(alltypes['double_col'], alltypes['double_col'],)
    result = expr.execute()
    expected = df.apply(
        lambda row: [row['double_col'], row['double_col']], axis=1
    )
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
# Issues #2370
@pytest.mark.xfail_backends(['bigquery'])
def test_array_concat(backend, con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr)
    assert result == [1, 2, 3, 2, 1]


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_array_length(backend, con):
    expr = ibis.literal([1, 2, 3]).length()
    assert con.execute(expr) == 3


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_list_literal(backend, con):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    result = con.execute(expr)
    assert np.array_equal(result, arr)


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_np_array_literal(backend, con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr)
    assert np.array_equal(result, arr)
