import pytest

import pandas as pd


def test_time():
    pass


def test_date():
    pass


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',  # 'W' TODO(kszucs): seems like numpy choses wednesday for W
    'h', 'm', 's', 'ms', 'us', 'ns'
])
def test_timestamp_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    with backend.skip_unsupported():
        result = expr.execute()

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',  # 'W' TODO(kszucs): seems like numpy choses wednesday for W
])
def test_date_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.date().truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    with backend.skip_unsupported():
        result = expr.execute()

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
