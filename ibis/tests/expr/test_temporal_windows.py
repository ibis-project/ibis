from __future__ import annotations

from operator import methodcaller

import pytest

import ibis
import ibis.common.exceptions as com
from ibis.common.deferred import _


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("tumble", size=ibis.interval(minutes=15)),
        methodcaller(
            "hop", size=ibis.interval(minutes=15), slide=ibis.interval(minutes=1)
        ),
    ],
    ids=["tumble", "hop"],
)
@pytest.mark.parametrize("by", ["g", _.g, ["g"]])
def test_window_by_agg_schema(table, method, by):
    expr = method(table.window_by(time_col=table.i))
    expr = expr.agg(by=by, a_sum=_.a.sum())
    expected_schema = ibis.schema(
        {
            "window_start": "timestamp",
            "window_end": "timestamp",
            "g": "string",
            "a_sum": "int64",
        }
    )
    assert expr.schema() == expected_schema


def test_window_by_with_non_timestamp_column(table):
    with pytest.raises(com.IbisInputError):
        table.window_by(time_col=table.a)


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("tumble", size=ibis.interval(minutes=15)),
        methodcaller(
            "hop", size=ibis.interval(minutes=15), slide=ibis.interval(minutes=1)
        ),
    ],
    ids=["tumble", "hop"],
)
@pytest.mark.parametrize("by", ["g", _.g, ["g"]])
def test_window_by_grouped_agg(table, method, by):
    expr = method(table.window_by(time_col=table.i))
    expr = expr.group_by(by).agg(a_sum=_.a.sum())
    expected_schema = ibis.schema(
        {
            "window_start": "timestamp",
            "window_end": "timestamp",
            "g": "string",
            "a_sum": "int64",
        }
    )
    assert expr.schema() == expected_schema


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("tumble", size=ibis.interval(minutes=15)),
        methodcaller(
            "hop", size=ibis.interval(minutes=15), slide=ibis.interval(minutes=1)
        ),
    ],
    ids=["tumble", "hop"],
)
def test_window_by_global_agg(table, method):
    expr = method(table.window_by(time_col=table.i))
    expr = expr.agg(a_sum=_.a.sum())
    expected_schema = ibis.schema(
        {
            "window_start": "timestamp",
            "window_end": "timestamp",
            "a_sum": "int64",
        }
    )
    assert expr.schema() == expected_schema
