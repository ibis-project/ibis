import numpy as np
import pytest

import ibis
import ibis.expr.types as ir


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_array_column(backend, alltypes, df):
    expr = ibis.array([alltypes['double_col'], alltypes['double_col']])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: np.array(
            [row['double_col'], row['double_col']], dtype=object
        ),
        axis=1,
    )
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_array_scalar(backend, con, alltypes, df):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr)
    expected = np.array([1.0, 2.0, 3.0])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


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
    expected = np.array([1, 2, 3, 2, 1])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


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

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)


@pytest.mark.xfail_unsupported
@pytest.mark.skip_missing_feature(
    ['supports_arrays', 'supports_arrays_outside_of_select']
)
def test_np_array_literal(backend, con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr)

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)
