from __future__ import annotations

import pytest

import ibis
from ibis import _


def test_tumble_window_by_grouped_agg(alltypes):
    t = alltypes
    expr = (
        t.window_by(t.timestamp_col)
        .tumble(size=ibis.interval(days=10))
        .agg(by=["string_col"], avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "string_col", "avg"]
    assert result.shape == (740, 4)


def test_tumble_window_by_ungrouped_agg(alltypes):
    t = alltypes
    expr = (
        t.window_by(t.timestamp_col)
        .tumble(size=ibis.interval(days=1))
        .agg(avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "avg"]
    assert result.shape == (730, 3)


def test_hop_window_by_grouped_agg(alltypes):
    t = alltypes
    expr = (
        t.window_by(t.timestamp_col)
        .hop(size=ibis.interval(days=10), slide=ibis.interval(days=10))
        .agg(by=["string_col"], avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "string_col", "avg"]
    assert result.shape == (740, 4)


def test_hop_window_by_ungrouped_agg(alltypes):
    t = alltypes
    expr = (
        t.window_by(t.timestamp_col)
        .hop(size=ibis.interval(days=1), slide=ibis.interval(days=1))
        .agg(avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "avg"]
    assert result.shape == (730, 3)
