import pytest

from ibis.backends.flink.compiler.core import translate


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_mutating_join(batting, awards_players, how, snapshot):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == 'NL'].drop('yearID', 'lgID')

    predicate = ['playerID']

    expr = left.join(right, predicate, how=how)
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")
