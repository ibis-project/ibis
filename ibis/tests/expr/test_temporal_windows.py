from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
from ibis.common.deferred import _


def test_window_by_agg_schema(table):
    expr = (
        table.window_by(time_col=table.i)
        .tumble(size=ibis.interval(minutes=15))
        .agg(by=["g"], a_sum=_.a.sum())
    )
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
        table.window_by(time_col=table.a).tumble(size=ibis.interval(minutes=15))
