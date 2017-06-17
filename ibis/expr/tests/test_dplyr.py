import os

import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis

from ibis.expr.dplyr import (
    X, Y,
    n,
    groupby,
    summarize,
    select,
    sift,
    head,
    sort_by,
    mutate,
    mean,
    sum,
    var,
    std,
    min,
    max,
    inner_join,
    left_join,
    right_join,
    outer_join,
    desc,
    do,
    semi_join,
    anti_join,
    # sample_n,
    # sample_frac,
)


@pytest.fixture(scope='module')
def df():
    path = os.environ.get('IBIS_DIAMONDS_CSV', 'diamonds.csv')
    return pd.read_csv(path, index_col=None)


@pytest.fixture(
    params=[
        ibis.postgres.connect(database='ibis_testing'),
        ibis.sqlite.connect(
            '/home/phillip/data/ibis-testing-data/ibis_testing.db'
        ),
        ibis.pandas.connect({'diamonds': df(), 'other_diamonds': df()})
    ]
)
def client(request):
    return request.param


@pytest.fixture
def diamonds(client):
    return client.table('diamonds').head(1000)


@pytest.fixture
def other_diamonds(client):
    return client.table('other_diamonds').head(1000)


def test_dplyr(diamonds):
    expected = diamonds[diamonds.price * diamonds.price / 2.0 >= 100]
    expected = expected.groupby('cut').aggregate([
        expected.carat.max().name('max_carat'),
        expected.carat.mean().name('mean_carat'),
        expected.carat.min().name('min_carat'),
        expected.x.count().name('n'),
        expected.carat.std().name('std_carat'),
        expected.carat.sum().name('sum_carat'),
        expected.carat.var().name('var_carat'),
    ])
    expected = expected.mutate(
        foo=expected.mean_carat,
        bar=expected.var_carat
    ).sort_by([ibis.desc('foo'), 'bar']).head()

    result = (
        diamonds >> sift(X.price * X.price / 2.0 >= 100)
                 >> groupby(X.cut)
                 >> summarize(
                     max_carat=max(X.carat),
                     mean_carat=mean(X.carat),
                     min_carat=min(X.carat),
                     n=n(X.x),
                     std_carat=std(X.carat),
                     sum_carat=sum(X.carat),
                     var_carat=var(X.carat),
                    )
                 >> mutate(foo=X.mean_carat, bar=X.var_carat)
                 >> sort_by(desc(X.foo), X.bar)
                 >> head(5)
    )
    assert result.equals(expected)
    tm.assert_frame_equal(expected.execute(), result >> do())


@pytest.mark.parametrize(
    'join_func',
    [
        inner_join,
        left_join,
        pytest.mark.xfail(right_join, raises=KeyError),
        outer_join,
        semi_join,
        anti_join,
    ]
)
def test_join(diamonds, other_diamonds, join_func):
    result = (
        diamonds >> join_func(other_diamonds, on=X.cut == Y.cut)
                 >> select(X.x, Y.y)
    )
    joined = getattr(diamonds, join_func.__name__)(
        other_diamonds, diamonds.cut == other_diamonds.cut
    )
    expected = joined[diamonds.x, other_diamonds.y]
    assert result.equals(expected)


@pytest.mark.parametrize(
    'column',
    [
        'carat',
        'cut',
        'color',
        'clarity',
        'depth',
        'table',
        'price',
        'x',
        'y',
        'z',
    ] + list(range(10))
)
def test_pull(diamonds, column):
    result = diamonds >> X[column]
    expected = diamonds[column]
    assert result.equals(expected)
    tm.assert_series_equal(expected.execute(), result >> do())


def test_do(diamonds):
    tm.assert_frame_equal(diamonds.execute(), diamonds >> do())


def test_simple_arithmetic(diamonds):
    result = diamonds >> mean(X.carat) + 1
    expected = diamonds.carat.mean() + 1
    assert result.equals(expected)
    assert float(expected.execute()) == float(result >> do())
