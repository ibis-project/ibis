import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    'column',
    [
        'string_col',
        'double_col',
        'date_string_col',
        pytest.param('timestamp_col', marks=pytest.mark.skip(reason='hangs')),
    ],
)
@pytest.mark.xfail_unsupported
def test_distinct_column(backend, alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)


@pytest.mark.parametrize(
    'backend_pseudocolumn', [{'omniscidb': 'rowid', 'sqlite': 'rowid'}]
)
@pytest.mark.xfail_unsupported
def test_pseudocolumn_rowid(con, backend, backend_pseudocolumn):
    # pseudocolumn needs to be used by a table expression directly
    # alltypes fixture from some backends maybe apply some operation on it
    t = con.table('functional_alltypes')
    backend_col_name = backend_pseudocolumn.get(backend.name, None)

    if not backend_col_name:
        pytest.xfail('No test defined for {}'.format(backend.name))

    row_id = t.row_id(backend_col_name)

    expr = t[[row_id]]
    result = expr.execute()
    expected = pd.Series(np.arange(len(result)))
    pd.testing.assert_series_equal(
        result.rowid, expected, check_names=False, check_dtype=False
    )


@pytest.mark.parametrize(
    'backend_pseudocolumn', [{'omniscidb': 'rowid', 'sqlite': 'rowid'}]
)
@pytest.mark.xfail_unsupported
def test_pseudocolumn_rowid_xfail(con, backend, backend_pseudocolumn):
    # pseudocolumn needs to be used by a table expression directly
    # alltypes fixture from some backends maybe apply some operation on it
    t = con.table('functional_alltypes')
    t = t.mutate(newcol=1)

    backend_col_name = backend_pseudocolumn.get(backend.name, None)

    if not backend_col_name:
        pytest.xfail('No test defined for {}'.format(backend.name))

    with pytest.raises(NotImplementedError):
        t.row_id(backend_col_name)
