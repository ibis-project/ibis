from ibis.backends.flink.compiler.core import translate


def test_mutating_join(batting, awards_players, snapshot):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == 'NL'].drop('yearID', 'lgID')

    predicate = ['playerID']

    expr = left.join(right, predicate)
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")
