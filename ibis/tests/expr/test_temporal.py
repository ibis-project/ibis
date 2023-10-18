from __future__ import annotations

import datetime
import operator

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.common.temporal import IntervalUnit
from ibis.expr import api
from ibis.tests.util import assert_equal


def test_temporal_literals():
    date = ibis.literal("2015-01-01", "date")
    assert isinstance(date, ir.DateScalar)

    timestamp = ibis.literal("2017-01-01 12:00:33", "timestamp")
    assert isinstance(timestamp, ir.TimestampScalar)


def test_interval_function_integers():
    # No args, default to 0 seconds
    assert_equal(ibis.interval(), ibis.interval(0, "s"))

    # Default unit is seconds
    assert_equal(ibis.interval(1), ibis.interval(1, "s"))

    # unit is used if provided
    res = ibis.interval(1, "D")
    sol = ibis.literal(1, type=dt.Interval("D"))
    assert_equal(res, sol)


def test_interval_function_timedelta():
    res = ibis.interval(datetime.timedelta())
    sol = ibis.interval()
    assert_equal(res, sol)

    res = ibis.interval(datetime.timedelta(microseconds=10))
    sol = ibis.interval(10, "us")
    assert_equal(res, sol)

    res = ibis.interval(datetime.timedelta(days=5))
    sol = ibis.interval(5, "D")
    assert_equal(res, sol)

    res = ibis.interval(datetime.timedelta(seconds=10, microseconds=2))
    sol = ibis.interval(2, "us") + ibis.interval(10, "s")
    assert_equal(res, sol)


@pytest.mark.parametrize(
    "kw,unit",
    [
        ("nanoseconds", "ns"),
        ("microseconds", "us"),
        ("milliseconds", "ms"),
        ("seconds", "s"),
        ("minutes", "m"),
        ("hours", "h"),
        ("days", "D"),
        ("weeks", "W"),
        ("months", "M"),
        ("quarters", "Q"),
        ("years", "Y"),
    ],
)
def test_interval_function_unit_keywords(kw, unit):
    res = ibis.interval(**{kw: 1})
    sol = ibis.interval(1, unit)
    assert_equal(res, sol)


def test_interval_function_multiple_keywords():
    res = ibis.interval(microseconds=10, hours=3, days=2)
    sol = ibis.interval(10, "us") + ibis.interval(3, "h") + ibis.interval(2, "D")
    assert_equal(res, sol)


def test_interval_function_invalid():
    with pytest.raises(TypeError, match="integer or timedelta"):
        ibis.interval(1.5)

    with pytest.raises(TypeError, match="'value' and 'microseconds'"):
        ibis.interval(1, microseconds=2)


@pytest.mark.parametrize(
    ("interval", "unit", "expected"),
    [
        (api.interval(months=3), "Q", api.interval(quarters=1)),
        (api.interval(months=12), "Y", api.interval(years=1)),
        (api.interval(quarters=8), "Y", api.interval(years=2)),
        (api.interval(days=14), "W", api.interval(weeks=2)),
        (api.interval(minutes=240), "h", api.interval(hours=4)),
        (api.interval(seconds=360), "m", api.interval(minutes=6)),
        (api.interval(milliseconds=5000), "s", api.interval(seconds=5)),
        (api.interval(microseconds=5000000), "s", api.interval(seconds=5)),
        (api.interval(nanoseconds=5000000000), "s", api.interval(seconds=5)),
        (api.interval(seconds=3 * 86400), "D", api.interval(days=3)),
    ],
)
def test_upconvert_interval(interval, unit, expected):
    result = interval.to_unit(unit)
    assert result.equals(expected)


@pytest.mark.parametrize("target", ["Y", "Q", "M"])
@pytest.mark.parametrize(
    "delta",
    [
        api.interval(weeks=1),
        api.interval(days=1),
        api.interval(hours=1),
        api.interval(minutes=1),
        api.interval(seconds=1),
        api.interval(milliseconds=1),
        api.interval(microseconds=1),
        api.interval(nanoseconds=1),
    ],
)
def test_cannot_upconvert(delta, target):
    with pytest.raises(ValueError):
        delta.to_unit(target)


@pytest.mark.parametrize(
    "expr",
    [
        api.interval(days=2) * 2,
        api.interval(days=2) * (-2),
        3 * api.interval(days=2),
        (-3) * api.interval(days=2),
    ],
)
def test_multiply(expr):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == IntervalUnit("D")


@pytest.mark.parametrize(
    ("expr", "expected_unit"),
    [
        (
            api.interval(days=1) + api.interval(days=1),
            IntervalUnit("D"),
        ),
        (
            api.interval(days=2) + api.interval(hours=4),
            IntervalUnit("h"),
        ),
        (
            api.interval(seconds=1) + ibis.interval(minutes=2),
            IntervalUnit("s"),
        ),
    ],
)
def test_add(expr, expected_unit):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == expected_unit


@pytest.mark.parametrize(
    ("expr", "expected_unit"),
    [
        (
            api.interval(days=3) - api.interval(days=1),
            IntervalUnit("D"),
        ),
        (
            api.interval(days=2) - api.interval(hours=4),
            IntervalUnit("h"),
        ),
        (
            api.interval(minutes=2) - api.interval(seconds=1),
            IntervalUnit("s"),
        ),
    ],
)
def test_subtract(expr, expected_unit):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == expected_unit


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (api.interval(seconds=2).to_unit("s"), api.interval(seconds=2)),
        (
            api.interval(seconds=2).to_unit("ms"),
            api.interval(milliseconds=2 * 1000),
        ),
        (
            api.interval(seconds=2).to_unit("us"),
            api.interval(microseconds=2 * 1000000),
        ),
        (
            api.interval(seconds=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 1000000000),
        ),
        (
            api.interval(milliseconds=2).to_unit("ms"),
            api.interval(milliseconds=2),
        ),
        (
            api.interval(milliseconds=2).to_unit("us"),
            api.interval(microseconds=2 * 1000),
        ),
        (
            api.interval(milliseconds=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 1000000),
        ),
        (
            api.interval(microseconds=2).to_unit("us"),
            api.interval(microseconds=2),
        ),
        (
            api.interval(microseconds=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 1000),
        ),
        (
            api.interval(nanoseconds=2).to_unit("ns"),
            api.interval(nanoseconds=2),
        ),
    ],
)
def test_downconvert_second_parts(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (api.interval(hours=2).to_unit("h"), api.interval(hours=2)),
        (api.interval(hours=2).to_unit("m"), api.interval(minutes=2 * 60)),
        (api.interval(hours=2).to_unit("s"), api.interval(seconds=2 * 3600)),
        (
            api.interval(hours=2).to_unit("ms"),
            api.interval(milliseconds=2 * 3600000),
        ),
        (
            api.interval(hours=2).to_unit("us"),
            api.interval(microseconds=2 * 3600000000),
        ),
        (
            api.interval(hours=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 3600000000000),
        ),
    ],
)
def test_downconvert_hours(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (api.interval(minutes=2).to_unit("m"), api.interval(minutes=2)),
        (api.interval(minutes=2).to_unit("s"), api.interval(seconds=2 * 60)),
        (
            api.interval(minutes=2).to_unit("ms"),
            api.interval(milliseconds=2 * 60 * 1000),
        ),
        (
            api.interval(minutes=2).to_unit("us"),
            api.interval(microseconds=2 * 60 * 1000000),
        ),
        (
            api.interval(minutes=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 60 * 1000000000),
        ),
        (api.interval(hours=2).to_unit("h"), api.interval(hours=2)),
        (api.interval(hours=2).to_unit("m"), api.interval(minutes=2 * 60)),
        (api.interval(hours=2).to_unit("s"), api.interval(seconds=2 * 3600)),
        (api.interval(weeks=2).to_unit("D"), api.interval(days=2 * 7)),
        (api.interval(days=2).to_unit("D"), api.interval(days=2)),
        (api.interval(years=2).to_unit("Y"), api.interval(years=2)),
        (api.interval(years=2).to_unit("M"), api.interval(months=24)),
        (api.interval(years=2).to_unit("Q"), api.interval(quarters=8)),
        (api.interval(months=2).to_unit("M"), api.interval(months=2)),
        (api.interval(weeks=2).to_unit("h"), api.interval(hours=2 * 7 * 24)),
        (api.interval(days=2).to_unit("h"), api.interval(hours=2 * 24)),
        (api.interval(days=2).to_unit("m"), api.interval(minutes=2 * 24 * 60)),
        (api.interval(days=2).to_unit("s"), api.interval(seconds=2 * 24 * 60 * 60)),
        (
            api.interval(days=2).to_unit("ms"),
            api.interval(milliseconds=2 * 24 * 60 * 60 * 1_000),
        ),
        (
            api.interval(days=2).to_unit("us"),
            api.interval(microseconds=2 * 24 * 60 * 60 * 1_000_000),
        ),
        (
            api.interval(days=2).to_unit("ns"),
            api.interval(nanoseconds=2 * 24 * 60 * 60 * 1_000_000_000),
        ),
    ],
)
def test_downconvert_interval(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ("a", "b", "unit"),
    [
        (api.interval(days=1), api.interval(days=3), "D"),
        (api.interval(seconds=1), api.interval(hours=10), "s"),
        (api.interval(hours=3), api.interval(days=2), "h"),
    ],
)
def test_combine_with_different_kinds(a, b, unit):
    assert (a + b).type().unit == IntervalUnit(unit)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (api.interval(quarters=2), api.interval(quarters=2)),
        (api.interval(weeks=2), api.interval(weeks=2)),
        (api.interval(days=3), api.interval(days=3)),
        (api.interval(hours=4), api.interval(hours=4)),
        (api.interval(minutes=5), api.interval(minutes=5)),
        (api.interval(seconds=6), api.interval(seconds=6)),
        (api.interval(milliseconds=7), api.interval(milliseconds=7)),
        (api.interval(microseconds=8), api.interval(microseconds=8)),
        (api.interval(nanoseconds=9), api.interval(nanoseconds=9)),
    ],
)
def test_timedelta_generic_api(case, expected):
    assert case.equals(expected)


ONE_DAY = ibis.interval(days=1)
ONE_HOUR = ibis.interval(hours=1)
ONE_SECOND = ibis.interval(seconds=1)


@pytest.mark.parametrize(
    ("func", "expected_op", "expected_type", "left", "right"),
    [
        # time
        param(
            operator.add,
            ops.TimeAdd,
            ir.TimeColumn,
            lambda _: ONE_HOUR,
            lambda t: t.k,
            id="time_add_interval_column",
        ),
        param(
            operator.add,
            ops.TimeAdd,
            ir.TimeColumn,
            lambda t: t.k,
            lambda _: ONE_HOUR,
            id="time_add_column_interval",
        ),
        param(
            operator.sub,
            ops.TimeSub,
            ir.TimeColumn,
            lambda _: ONE_HOUR,
            lambda t: t.k,
            id="time_sub_interval_column",
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="interval - time is not yet implemented",
            ),
        ),
        param(
            operator.sub,
            ops.TimeSub,
            ir.TimeColumn,
            lambda t: t.k,
            lambda _: ONE_HOUR,
            id="time_sub_column_interval",
        ),
        # date
        param(
            operator.add,
            ops.DateAdd,
            ir.DateColumn,
            lambda _: ONE_DAY,
            lambda t: t.j,
            id="date_add_interval_column",
        ),
        param(
            operator.add,
            ops.DateAdd,
            ir.DateColumn,
            lambda t: t.j,
            lambda _: ONE_DAY,
            id="date_add_column_interval",
        ),
        param(
            operator.sub,
            ops.DateSub,
            ir.DateColumn,
            lambda _: ONE_DAY,
            lambda t: t.j,
            id="date_sub_interval_column",
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="interval - date is not yet implemented",
            ),
        ),
        param(
            operator.sub,
            ops.DateSub,
            ir.DateColumn,
            lambda t: t.j,
            lambda _: ONE_DAY,
            id="date_sub_column_interval",
        ),
        # timestamp
        param(
            operator.add,
            ops.TimestampAdd,
            ir.TimestampColumn,
            lambda _: ONE_SECOND,
            lambda t: t.i,
            id="timestamp_add_interval_column",
        ),
        param(
            operator.add,
            ops.TimestampAdd,
            ir.TimestampColumn,
            lambda t: t.i,
            lambda _: ONE_SECOND,
            id="timestamp_add_column_interval",
        ),
        param(
            operator.sub,
            ops.TimestampSub,
            ir.TimestampColumn,
            lambda _: ONE_SECOND,
            lambda t: t.i,
            id="timestamp_sub_interval_column",
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="interval - timestamp is not yet implemented",
            ),
        ),
        param(
            operator.sub,
            ops.TimestampSub,
            ir.TimestampColumn,
            lambda t: t.i,
            lambda _: ONE_SECOND,
            id="timestamp_sub_column_interval",
        ),
    ],
)
def test_interval_arithmetic(
    table,
    func,
    expected_op,
    expected_type,
    left,
    right,
):
    lhs = left(table)
    rhs = right(table)
    expr = func(lhs, rhs)
    assert isinstance(expr, expected_type)
    assert isinstance(expr.op(), expected_op)


@pytest.mark.parametrize(
    "literal",
    [
        api.interval(3600),
        api.interval(datetime.timedelta(days=3)),
        api.interval(years=1),
        api.interval(quarters=3),
        api.interval(months=2),
        api.interval(weeks=3),
        api.interval(days=-1),
        api.interval(hours=-3),
        api.interval(minutes=5),
        api.interval(seconds=-8),
    ],
)
def test_interval(literal):
    assert isinstance(literal, ir.IntervalScalar)
    repr(literal)  # repr() must return in a reasonable amount of time


def test_timestamp_arithmetics():
    ts1 = api.timestamp(datetime.datetime.now())
    ts2 = api.timestamp(datetime.datetime.today())

    i1 = api.interval(minutes=30)

    # TODO: raise for unsupported operations too
    for expr in [ts2 - ts1, ts1 - ts2]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimestampDiff)
        assert expr.type() == dt.Interval("s")

    for expr in [ts1 - i1, ts2 - i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampSub)

    for expr in [ts1 + i1, ts2 + i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampAdd)


def test_date_arithmetics():
    d1 = api.date("2015-01-02")
    d2 = api.date("2017-01-01")
    i1 = api.interval(weeks=3)

    for expr in [d1 - d2, d2 - d1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.DateDiff)
        assert expr.type() == dt.Interval("D")

    for expr in [d1 - i1, d2 - i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateSub)

    for expr in [d1 + i1, d2 + i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateAdd)


def test_time_arithmetics():
    t1 = api.time("18:00")
    t2 = api.time("19:12")
    i1 = api.interval(minutes=3)

    for expr in [t1 - t2, t2 - t1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimeDiff)
        assert expr.type() == dt.Interval("s")

    for expr in [t1 - i1, t2 - i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeSub)

    for expr in [t1 + i1, t2 + i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeAdd)


def test_invalid_date_arithmetics():
    d1 = api.date("2015-01-02")
    i1 = api.interval(seconds=300)
    i2 = api.interval(minutes=15)
    i3 = api.interval(hours=1)

    for i in [i1, i2, i3]:
        with pytest.raises(TypeError):
            d1 - i
        with pytest.raises(TypeError):
            d1 + i


@pytest.mark.parametrize(
    ("value", "prop", "expected_unit"),
    [
        (api.interval(seconds=3600), "nanoseconds", "ns"),
        (api.interval(seconds=3600), "microseconds", "us"),
        (api.interval(seconds=3600), "milliseconds", "ms"),
        (api.interval(seconds=3600), "seconds", "s"),
        (api.interval(seconds=3600), "minutes", "m"),
        (api.interval(seconds=3600), "hours", "h"),
        (api.interval(weeks=3), "days", "D"),
        (api.interval(days=21), "weeks", "W"),
    ],
)
def test_interval_properties(value, prop, expected_unit):
    assert getattr(value, prop).type().unit == IntervalUnit(expected_unit)


@pytest.mark.parametrize(
    "interval",
    [api.interval(years=1), api.interval(quarters=4), api.interval(months=12)],
)
@pytest.mark.parametrize(
    ("prop", "expected_unit"),
    [("months", "M"), ("quarters", "Q"), ("years", "Y")],
)
def test_interval_monthly_properties(interval, prop, expected_unit):
    assert getattr(interval, prop).type().unit == IntervalUnit(expected_unit)


@pytest.mark.parametrize(
    ("interval", "prop"),
    [
        (api.interval(hours=48), "months"),
        (api.interval(years=2), "seconds"),
        (api.interval(quarters=1), "weeks"),
    ],
)
def test_unsupported_properties(interval, prop):
    with pytest.raises(ValueError):
        getattr(interval, prop)


@pytest.mark.parametrize("column", ["a", "b", "c", "d"])  # integer columns
@pytest.mark.parametrize(
    "unit", ["Y", "Q", "M", "D", "W", "h", "m", "s", "ms", "us", "ns"]
)
def test_integer_to_interval(column, unit, table):
    c = table[column]
    i = c.to_interval(unit)
    assert isinstance(i, ir.IntervalColumn)
    assert i.type().unit == IntervalUnit(unit)


@pytest.mark.parametrize(
    "unit", ["Y", "Q", "M", "D", "W", "h", "m", "s", "ms", "us", "ns"]
)
@pytest.mark.parametrize(
    "operands",
    [
        lambda t, u: (api.interval(3, unit=u), api.interval(2, unit=u)),
        lambda t, u: (api.interval(3, unit=u), api.interval(3, unit=u)),
        lambda t, u: (t.c.to_interval(unit=u), api.interval(2, unit=u)),
        lambda t, u: (t.c.to_interval(unit=u), t.d.to_interval(unit=u)),
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ],
    ids=lambda op: op.__name__,
)
def test_interval_comparisons(unit, operands, operator, table):
    a, b = operands(table, unit)
    expr = operator(a, b)

    assert isinstance(a, ir.IntervalValue)
    assert isinstance(b, ir.IntervalValue)
    assert isinstance(expr, ir.BooleanValue)


@pytest.mark.parametrize(
    "operands",
    [
        lambda t: (api.date("2016-01-01"), api.date("2016-02-02")),
        lambda t: (t.j, api.date("2016-01-01")),
        lambda t: (api.date("2016-01-01"), t.j),
        lambda t: (t.j, t.i.date()),
        lambda t: (t.i.date(), t.j),
    ],
    ids=[
        "literal-literal",
        "column-literal",
        "literal-column",
        "column-casted",
        "casted-column",
    ],
)
@pytest.mark.parametrize(
    "interval",
    [
        lambda t: api.interval(years=4),
        lambda t: api.interval(quarters=4),
        lambda t: api.interval(months=3),
        lambda t: api.interval(weeks=2),
        lambda t: api.interval(days=1),
        lambda t: t.c.to_interval(unit="Y"),
        lambda t: t.c.to_interval(unit="M"),
        lambda t: t.c.to_interval(unit="W"),
        lambda t: t.c.to_interval(unit="D"),
    ],
    ids=[
        "years",
        "quarters",
        "months",
        "weeks",
        "days",
        "to-years",
        "to-months",
        "to-weeks",
        "to-days",
    ],
)
@pytest.mark.parametrize(
    "arithmetic",
    [lambda a, i: a - i, lambda a, i: a + i, lambda a, i: i + a],
    ids=["subtract", "radd", "add"],
)
@pytest.mark.parametrize(
    "operator",
    [
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ],
    ids=lambda op: op.__name__,
)
def test_complex_date_comparisons(operands, interval, arithmetic, operator, table):
    (a, b), i = operands(table), interval(table)

    a_ = arithmetic(a, i)
    expr = operator(a_, b)

    assert isinstance(a, ir.DateValue)
    assert isinstance(b, ir.DateValue)
    assert isinstance(a_, ir.DateValue)
    assert isinstance(i, ir.IntervalValue)
    assert isinstance(expr, ir.BooleanValue)


def test_interval_column_name(table):
    c = table.i
    expr = (c - c).name("foo")
    assert expr.get_name() == "foo"


@pytest.mark.parametrize(
    "operand",
    [lambda t: api.timestamp(datetime.datetime.now()), lambda t: t.i],
    ids=["column", "literal"],
)
@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "y",
        "year",
        "YEAR",
        "YYYY",
        "SYYYY",
        "YYY",
        "YY",
        "Q",
        "q",
        "quarter",
        "QUARTER",
        "M",
        "month",
        "MONTH",
        "w",
        "W",
        "week",
        "WEEK",
        "d",
        "J",
        "day",
        "DAY",
        "h",
        "H",
        "HH24",
        "hour",
        "HOUR",
        "m",
        "MI",
        "minute",
        "MINUTE",
        "s",
        "second",
        "SECOND",
        "ms",
        "millisecond",
        "MILLISECOND",
        "us",
        "microsecond",
        "MICROSECOND",
        "ns",
        "nanosecond",
        "NANOSECOND",
    ],
)
def test_timestamp_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.TimestampValue)
    assert isinstance(expr.op(), ops.TimestampTruncate)


@pytest.mark.parametrize("operand", [lambda t: api.date("2018-01-01"), lambda t: t.j])
@pytest.mark.parametrize("unit", ["Y", "Q", "M", "D", "W"])
def test_date_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.DateValue)
    assert isinstance(expr.op(), ops.DateTruncate)


@pytest.mark.parametrize("operand", [lambda t: api.time("18:00"), lambda t: t.k])
@pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "us", "ns"])
def test_time_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.TimeValue)
    assert isinstance(expr.op(), ops.TimeTruncate)


def test_date_literal():
    expr = ibis.date(2022, 2, 4)
    sol = ops.DateFromYMD(2022, 2, 4).to_expr()
    assert expr.equals(sol)

    expr = ibis.date("2022-02-04")
    sol = ibis.literal("2022-02-04", type=dt.date)
    assert expr.equals(sol)


def test_date_expression():
    t = ibis.table({"x": "int", "y": "int", "z": "int", "s": "string"})
    deferred = ibis.date(_.x, _.y, _.z)
    expr = ibis.date(t.x, t.y, t.z)
    assert isinstance(expr.op(), ops.DateFromYMD)
    assert deferred.resolve(t).equals(expr)
    assert repr(deferred) == "date(_.x, _.y, _.z)"

    deferred = ibis.date(_.s)
    expr = ibis.date(t.s)
    assert deferred.resolve(t).equals(expr)
    assert repr(deferred) == "date(_.s)"


def test_time_literal():
    expr = ibis.time(1, 2, 3)
    sol = ops.TimeFromHMS(1, 2, 3).to_expr()
    assert expr.equals(sol)

    expr = ibis.time("01:02:03")
    sol = ibis.literal("01:02:03", type=dt.time)
    assert expr.equals(sol)


def test_time_expression():
    t = ibis.table({"x": "int", "y": "int", "z": "int", "s": "string"})
    deferred = ibis.time(_.x, _.y, _.z)
    expr = ibis.time(t.x, t.y, t.z)
    assert isinstance(expr.op(), ops.TimeFromHMS)
    assert deferred.resolve(t).equals(expr)
    assert repr(deferred) == "time(_.x, _.y, _.z)"

    deferred = ibis.time(_.s)
    expr = ibis.time(t.s)
    assert deferred.resolve(t).equals(expr)
    assert repr(deferred) == "time(_.s)"


def test_timestamp_literals():
    assert ibis.timestamp(2022, 2, 4, 16, 20, 00).type() == dt.timestamp


def test_timestamp_literal():
    expr = ibis.timestamp(2022, 2, 4, 16, 20, 0)
    sol = ops.TimestampFromYMDHMS(2022, 2, 4, 16, 20, 0).to_expr()
    assert expr.equals(sol)

    expr = ibis.timestamp("2022-02-04T01:02:03")
    sol = ibis.literal("2022-02-04T01:02:03", type=dt.timestamp)
    assert expr.equals(sol)

    expr = ibis.timestamp("2022-02-04T01:02:03Z")
    sol = ibis.literal("2022-02-04T01:02:03", type=dt.Timestamp(timezone="UTC"))
    assert expr.equals(sol)


def test_timestamp_expression():
    t = ibis.table(dict.fromkeys("abcdef", "int"))
    deferred = ibis.timestamp(_.a, _.b, _.c, _.d, _.e, _.f)
    expr = ibis.timestamp(t.a, t.b, t.c, t.d, t.e, t.f)
    assert isinstance(expr.op(), ops.TimestampFromYMDHMS)
    assert deferred.resolve(t).equals(expr)
    assert repr(deferred) == "timestamp(_.a, _.b, _.c, _.d, _.e, _.f)"

    t2 = ibis.table({"s": "string"})
    deferred = ibis.timestamp(_.s, timezone="UTC")
    expr = ibis.timestamp(t2.s, timezone="UTC")
    assert deferred.resolve(t2).equals(expr)
    assert repr(deferred) == "timestamp(_.s, timezone='UTC')"


def test_timestamp_bucket():
    ts = ibis.table({"ts": "timestamp"}).ts

    components = [
        "nanoseconds",
        "microseconds",
        "milliseconds",
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "quarters",
        "years",
    ]
    for component in components:
        kws = {component: 2}
        expr1 = ts.bucket(**kws)
        expr2 = ts.bucket(ibis.interval(**kws))
        assert expr1.equals(expr2)

    with pytest.raises(
        ValueError, match="Must specify either interval value or components"
    ):
        ts.bucket(ibis.interval(seconds=1), minutes=2)

    with pytest.raises(
        ValueError, match="Must specify either interval value or components"
    ):
        ts.bucket()
