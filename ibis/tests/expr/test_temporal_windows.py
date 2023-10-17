from __future__ import annotations

import datetime

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import selectors as s
from ibis.common.annotations import ValidationError
from ibis.common.deferred import _


def test_tumble_tvf_schema(schema, table):
    expr = table.window_by(time_col=table.i).tumble(
        window_size=ibis.interval(minutes=15)
    )
    expected_schema = ibis.schema(
        schema
        + [
            ("window_start", dt.Timestamp(scale=3)),
            ("window_end", dt.Timestamp(scale=3)),
            ("window_time", dt.Timestamp(scale=3)),
        ]
    )
    assert expr.schema() == expected_schema


@pytest.mark.parametrize("wrong_type_window_size", ["60", 60])
def test_create_tumble_tvf_with_wrong_scalar_type(table, wrong_type_window_size):
    with pytest.raises(ValidationError, match=".* is not coercible to a .*"):
        table.window_by(time_col=table.i).tumble(window_size=wrong_type_window_size)


def test_create_tumble_tvf_with_nonexistent_time_col(table):
    with pytest.raises(com.IbisTypeError, match="Column .* is not found in table"):
        table.window_by(time_col=table["nonexistent"]).tumble(
            window_size=datetime.timedelta(seconds=60)
        )


def test_create_tumble_tvf_with_nonscalar_window_size(schema):
    schema.append(("l", "interval"))
    table = ibis.table(schema, name="table")
    with pytest.raises(ValidationError, match=".* is not coercible to a .*"):
        table.window_by(time_col=table.i).tumble(window_size=table.l)


def test_create_tumble_tvf_with_non_timestamp_time_col(table):
    with pytest.raises(ValidationError, match=".* is not coercible to a .*"):
        table.window_by(time_col=table.e).tumble(window_size=ibis.interval(minutes=15))


def test_create_tumble_tvf_with_str_time_col(table):
    expr = table.window_by(time_col="i").tumble(window_size=ibis.interval(minutes=15))
    assert isinstance(expr.op(), ops.TumbleWindowingTVF)
    assert expr.op().time_col == table.i.op()


@pytest.mark.parametrize("deferred", [_["i"], _.i])
def test_create_tumble_tvf_with_deferred_time_col(table, deferred):
    expr = table.window_by(time_col=deferred.resolve(table)).tumble(
        window_size=ibis.interval(minutes=15)
    )
    assert isinstance(expr.op(), ops.TumbleWindowingTVF)
    assert expr.op().time_col == table.i.op()


def test_create_tumble_tvf_with_selector_time_col(table):
    expr = table.window_by(time_col=s.c("i")).tumble(
        window_size=ibis.interval(minutes=15)
    )
    assert isinstance(expr.op(), ops.TumbleWindowingTVF)
    assert expr.op().time_col == table.i.op()
