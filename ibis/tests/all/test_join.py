import pytest

import pandas as pd


@pytest.fixture(scope='module')
def left(batting):
    return batting[batting.yearID == 2015]


@pytest.fixture(scope='module')
def right(awards_players):
    return awards_players[awards_players.lgID == 'NL']


@pytest.fixture(scope='module')
def left_df(left):
    return left.execute()


@pytest.fixture(scope='module')
def right_df(right):
    return right.execute()


@pytest.mark.skip
@pytest.mark.parametrize('how', [
    'inner',
    'left',
    'right',
    'outer',
    pytest.mark.xfail(
        'semi',
        raises=NotImplementedError,
        reason='Semi join not implemented'
    ),
    pytest.mark.xfail(
        'anti',
        raises=NotImplementedError,
        reason='Anti join not implemented'
    ),
])
def test_join_project_left_table(backend, con, left, right,
                                 left_df, right_df, how):
    predicate = ['playerID']
    expr = left.join(right, predicate, how=how)[left]

    with backend.skip_unsupported():
        result = expr.execute()

    joined = pd.merge(left_df, right_df, how=how, on=predicate,
                      suffixes=('', '_y'))
    expected = joined[list(left.columns)]

    backend.assert_frame_equal(result[expected.columns], expected,
                               check_like=True)
