import decimal

import pytest

import pandas as pd

import ibis
import ibis.expr.datatypes as dt


@pytest.fixture
def df():
    return pd.DataFrame({
        'plain_int64': list(range(1, 4)),
        'plain_strings': list('abc'),
        'plain_float64': [4.0, 5.0, 6.0],
        'plain_datetimes_naive': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ),
        'plain_datetimes_ny': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ).dt.tz_localize('America/New_York'),
        'plain_datetimes_utc': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ).dt.tz_localize('UTC'),
        'dup_strings': list('dad'),
        'dup_ints': [1, 2, 1],
        'float64_as_strings': ['100.01', '234.23', '-999.34'],
        'int64_as_strings': list(map(str, range(1, 4))),
        'strings_with_space': [' ', 'abab', 'ddeeffgg'],
        'int64_with_zeros': [0, 1, 0],
        'float64_with_zeros': [1.0, 0.0, 1.0],
        'strings_with_nulls': ['a', None, 'b'],
        'datetime_strings_naive': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ).astype(str),
        'datetime_strings_ny': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ).dt.tz_localize('America/New_York').astype(str),
        'datetime_strings_utc': pd.Series(
            pd.date_range(start='2017-01-02 01:02:03.234', periods=3).values,
        ).dt.tz_localize('UTC').astype(str),
        'decimal': list(map(decimal.Decimal, ['1.0', '2', '3.234'])),
        'array_of_float64': [[1.0, 2.0], [3.0], []],
        'array_of_int64': [[1, 2], [], [3]],
        'array_of_strings': [['a', 'b'], [], ['c']],
    })


@pytest.fixture
def df1():
    return pd.DataFrame(
        {'key': list('abcd'), 'value': [3, 4, 5, 6], 'key2': list('eeff')}
    )


@pytest.fixture
def df2():
    return pd.DataFrame(
        {'key': list('ac'), 'other_value': [4.0, 6.0], 'key3': list('fe')}
    )


@pytest.fixture
def client(df, df1, df2):
    return ibis.pandas.connect(
        {'df': df, 'df1': df1, 'df2': df2, 'left': df1, 'right': df2}
    )


@pytest.fixture
def t(client):
    return client.table(
        'df',
        schema={
            'decimal': dt.Decimal(4, 3),
            'array_of_float64': dt.Array(dt.double),
            'array_of_int64': dt.Array(dt.int64),
            'array_of_strings': dt.Array(dt.string),
        }
    )


@pytest.fixture
def left(client):
    return client.table('left')


@pytest.fixture
def right(client):
    return client.table('right')
