import pytest

# import pandas as pd


def test_time():
    pass


def test_date():
    pass


@pytest.mark.parametrize(('expr_fn', 'expected_fn'), [
    (lambda t: t.timestamp_col.truncate('d'),
     lambda t: t.timestamp_col.dt.floor('d')),
    (lambda t: t.timestamp_col.truncate('h'),
     lambda t: t.timestamp_col.dt.floor('h')),
    (lambda t: t.timestamp_col.truncate('s'),
     lambda t: t.timestamp_col.dt.floor('s')),

])
def test_timestamp_truncate(backend, alltypes, df, expr_fn, expected_fn):
    if backend.name == 'sqlite':
        pytest.skip('')

    expr = expr_fn(alltypes)
    expected = expected_fn(df)

    with backend.skip_unsupported():
        result = expr.execute()

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


def test_interval():
    pass
