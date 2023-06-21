from ibis.backends.flink.compiler.core import translate


def test_join(batting, awards_players, snapshot):
    left = batting
    right = awards_players

    predicate = ['playerID']

    expr = left.join(right, predicate)
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")
