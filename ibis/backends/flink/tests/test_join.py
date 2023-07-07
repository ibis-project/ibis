from __future__ import annotations

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


def test_join_then_filter_no_column_overlap(awards_players, batting, snapshot):
    left = batting[batting.yearID == 2015]
    year = left.yearID.name("year")
    left = left[year, "RBI"]
    right = awards_players[awards_players.lgID == 'NL']

    expr = left.join(right, left.year == right.yearID)
    filters = [expr.RBI == 9]
    q = expr.filter(filters)
    result = translate(q.as_table().op())
    snapshot.assert_match(str(result), "out.sql")


def test_mutate_then_join_no_column_overlap(batting, awards_players, snapshot):
    left = batting.mutate(year=batting.yearID).filter(lambda t: t.year == 2015)
    left = left["year", "RBI"]
    right = awards_players
    expr = left.join(right, left.year == right.yearID)
    result = translate(expr.limit(5).as_table().op())
    snapshot.assert_match(str(result), "out.sql")
