import pandas as pd
import pytest
from pytest import param

from ibis.tests.backends import Pandas, PySpark

# add here backends that passes join tests
all_db_join_supported = [Pandas, PySpark]


@pytest.fixture(scope='module')
def left(batting):
    return batting[batting.yearID == 2015]


@pytest.fixture(scope='module')
def right(awards_players):
    return awards_players[awards_players.lgID == 'NL'].drop(['yearID', 'lgID'])


@pytest.fixture(scope='module')
def left_df(left):
    return left.execute()


@pytest.fixture(scope='module')
def right_df(right):
    return right.execute()


@pytest.mark.parametrize(
    'how',
    [
        'inner',
        'left',
        'right',
        'outer',
        param(
            'semi',
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason='Semi join not implemented'
            ),
        ),
        param(
            'anti',
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason='Anti join not implemented'
            ),
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_join_supported)
@pytest.mark.xfail_unsupported
def test_join_project_left_table(
    backend, con, left, right, left_df, right_df, how
):
    predicate = ['playerID']
    result_order = ['playerID', 'yearID', 'lgID', 'stint']
    expr = left.join(right, predicate, how=how)[left]
    result = expr.execute().sort_values(result_order)

    joined = pd.merge(
        left_df, right_df, how=how, on=predicate, suffixes=('', '_y')
    ).sort_values(result_order)
    expected = joined[list(left.columns)]

    backend.assert_frame_equal(
        result[expected.columns], expected, check_like=True
    )
