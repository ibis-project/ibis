from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr import api


def test_field_select(alltypes):
    assert isinstance(alltypes.i, ir.TimestampColumn)


def test_string_cast_to_timestamp(alltypes):
    casted = alltypes.g.cast("timestamp")
    assert isinstance(casted, ir.TimestampColumn)

    string = api.literal("2000-01-01")
    casted = string.cast("timestamp")
    assert isinstance(casted, ir.TimestampScalar)


@pytest.mark.parametrize(
    ("field", "expected_operation", "expected_type"),
    [
        ("year", ops.ExtractYear, ir.IntegerColumn),
        ("month", ops.ExtractMonth, ir.IntegerColumn),
        ("day", ops.ExtractDay, ir.IntegerColumn),
        ("hour", ops.ExtractHour, ir.IntegerColumn),
        ("minute", ops.ExtractMinute, ir.IntegerColumn),
        ("second", ops.ExtractSecond, ir.IntegerColumn),
        ("microsecond", ops.ExtractMicrosecond, ir.IntegerColumn),
        ("millisecond", ops.ExtractMillisecond, ir.IntegerColumn),
    ],
)
def test_extract_fields(field, expected_operation, expected_type, alltypes):
    # type-size may be database specific
    result = getattr(alltypes.i, field)()
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


def test_now():
    result = api.now()
    assert isinstance(result, ir.TimestampScalar)
    assert isinstance(result.op(), ops.TimestampNow)


def test_timestamp_literals():
    expr = ibis.timestamp("2015-01-01 00:00:00")
    assert isinstance(expr, ir.TimestampScalar)


def test_invalid_timestamp_literal():
    with pytest.raises(TypeError, match="Unable to normalize"):
        ibis.timestamp("2015-01-01 00:71")


def test_comparison_timestamp(alltypes):
    expr = alltypes.i > alltypes.i.min() + ibis.interval(days=3)
    assert isinstance(expr, ir.BooleanColumn)


def test_comparisons_string(alltypes):
    val = "2015-01-01 00:00:00"
    expr = alltypes.i > val
    op = expr.op()
    assert op.right.dtype is dt.string

    expr2 = val < alltypes.i
    op = expr2.op()
    assert isinstance(op, ops.Greater)
    assert op.right.dtype is dt.string


def test_comparisons_pandas_timestamp(alltypes):
    pd = pytest.importorskip("pandas")

    val = pd.Timestamp("2015-01-01 00:00:00")
    expr = alltypes.i > val
    op = expr.op()
    assert isinstance(op, ops.Greater)
    assert isinstance(op.right, ops.Literal)
    assert isinstance(op.right.dtype, dt.Timestamp)

    expr = ibis.literal(val) < alltypes.i
    op = expr.op()
    assert isinstance(op, ops.Less)
    assert isinstance(op.left, ops.Literal)
    assert isinstance(op.left.dtype, dt.Timestamp)

    expr = val < alltypes.i
    op = expr.op()
    assert isinstance(op, ops.Greater)
    assert isinstance(op.right, ops.Literal)
    assert isinstance(op.right.dtype, dt.Timestamp)


def test_timestamp_precedence():
    ts = ibis.literal(datetime.now())
    highest_type = rlz.highest_precedence_dtype([ibis.null().op(), ts.op()])
    assert highest_type == dt.timestamp


@pytest.mark.parametrize(
    ("field", "expected_operation", "expected_type"),
    [
        ("year", ops.ExtractYear, ir.IntegerColumn),
        ("month", ops.ExtractMonth, ir.IntegerColumn),
        ("day", ops.ExtractDay, ir.IntegerColumn),
    ],
)
def test_timestamp_field_access_on_date(
    field, expected_operation, expected_type, alltypes
):
    date_col = alltypes.i.date()
    result = getattr(date_col, field)()
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


@pytest.mark.parametrize(
    "field", ["hour", "minute", "second", "microsecond", "millisecond"]
)
def test_timestamp_field_access_on_date_failure(field, alltypes):
    time_col = alltypes.i.date()
    with pytest.raises(AttributeError):
        getattr(time_col, field)


@pytest.mark.parametrize(
    ("field", "expected_operation", "expected_type"),
    [
        ("hour", ops.ExtractHour, ir.IntegerColumn),
        ("minute", ops.ExtractMinute, ir.IntegerColumn),
        ("second", ops.ExtractSecond, ir.IntegerColumn),
        ("microsecond", ops.ExtractMicrosecond, ir.IntegerColumn),
        ("millisecond", ops.ExtractMillisecond, ir.IntegerColumn),
    ],
)
def test_timestamp_field_access_on_time(
    field, expected_operation, expected_type, alltypes
):
    time_col = alltypes.i.time()
    result = getattr(time_col, field)()
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


@pytest.mark.parametrize("field", ["year", "month", "day"])
def test_timestamp_field_access_on_time_failure(field, alltypes):
    date_col = alltypes.i.time()
    with pytest.raises(AttributeError):
        getattr(date_col, field)


def test_integer_timestamp_fails():
    with pytest.raises(TypeError, match=r"Use ibis\.literal\(\.\.\.\)\.as_timestamp"):
        ibis.timestamp(42)


@pytest.mark.parametrize(
    "start",
    [
        "2002-01-01 00:00:00",
        datetime(2002, 1, 1, 0, 0, 0),
        ibis.timestamp("2002-01-01 00:00:00"),
        ibis.timestamp(datetime(2002, 1, 1, 0, 0, 0)),
        ibis.table({"start": "timestamp"}).start,
    ],
)
@pytest.mark.parametrize(
    "stop",
    [
        "2002-01-02 00:00:00",
        datetime(2002, 1, 2, 0, 0, 0),
        ibis.timestamp("2002-01-02 00:00:00"),
        ibis.timestamp(datetime(2002, 1, 2, 0, 0, 0)),
        ibis.table({"stop": "timestamp"}).stop,
    ],
)
@pytest.mark.parametrize("step", [ibis.interval(seconds=1), timedelta(seconds=1)])
def test_timestamp_range_with_str_inputs(start, stop, step):
    expr = ibis.range(start, stop, step)

    op = expr.op()

    assert op.start.dtype.is_timestamp()
    assert op.stop.dtype.is_timestamp()
    assert op.step.dtype.is_interval()

    dtype = expr.type()

    assert dtype.is_array()
    assert dtype.value_type.is_timestamp()
