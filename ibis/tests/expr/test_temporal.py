import datetime
import operator

import pytest

import ibis
import ibis.expr.api as api
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir


def test_temporal_literals():
    date = ibis.literal('2015-01-01', 'date')
    assert isinstance(date, ir.DateScalar)

    timestamp = ibis.literal('2017-01-01 12:00:33', 'timestamp')
    assert isinstance(timestamp, ir.TimestampScalar)


@pytest.mark.parametrize(
    ('interval', 'unit', 'expected'),
    [
        (api.interval(months=3), 'Q', api.interval(quarters=1)),
        (api.interval(months=12), 'Y', api.interval(years=1)),
        (api.interval(quarters=8), 'Y', api.interval(years=2)),
        (api.interval(days=14), 'W', api.interval(weeks=2)),
        (api.interval(minutes=240), 'h', api.interval(hours=4)),
        (api.interval(seconds=360), 'm', api.interval(minutes=6)),
        (api.interval(seconds=3 * 86400), 'D', api.interval(days=3)),
        (api.interval(milliseconds=5000), 's', api.interval(seconds=5)),
        (api.interval(microseconds=5000000), 's', api.interval(seconds=5)),
        (api.interval(nanoseconds=5000000000), 's', api.interval(seconds=5)),
    ],
)
def test_upconvert(interval, unit, expected):
    result = interval.to_unit(unit)

    assert isinstance(result, ir.IntervalScalar)
    assert result.type().unit == expected.type().unit


@pytest.mark.parametrize('target', ['Y', 'Q', 'M'])
@pytest.mark.parametrize(
    'delta',
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
    'expr',
    [
        api.interval(days=2) * 2,
        api.interval(days=2) * (-2),
        3 * api.interval(days=2),
        (-3) * api.interval(days=2),
    ],
)
def test_multiply(expr):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == 'D'


@pytest.mark.parametrize(
    'expr',
    [
        api.interval(days=1) + api.interval(days=1),
        api.interval(days=2) + api.interval(hours=4),
    ],
)
def test_add(expr):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == 'D'


@pytest.mark.parametrize(
    'expr',
    [
        api.interval(days=3) - api.interval(days=1),
        api.interval(days=2) - api.interval(hours=4),
    ],
)
def test_subtract(expr):
    assert isinstance(expr, ir.IntervalScalar)
    assert expr.type().unit == 'D'


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (api.interval(seconds=2).to_unit('s'), api.interval(seconds=2)),
        (
            api.interval(seconds=2).to_unit('ms'),
            api.interval(milliseconds=2 * 1000),
        ),
        (
            api.interval(seconds=2).to_unit('us'),
            api.interval(microseconds=2 * 1000000),
        ),
        (
            api.interval(seconds=2).to_unit('ns'),
            api.interval(nanoseconds=2 * 1000000000),
        ),
        (
            api.interval(milliseconds=2).to_unit('ms'),
            api.interval(milliseconds=2),
        ),
        (
            api.interval(milliseconds=2).to_unit('us'),
            api.interval(microseconds=2 * 1000),
        ),
        (
            api.interval(milliseconds=2).to_unit('ns'),
            api.interval(nanoseconds=2 * 1000000),
        ),
        (
            api.interval(microseconds=2).to_unit('us'),
            api.interval(microseconds=2),
        ),
        (
            api.interval(microseconds=2).to_unit('ns'),
            api.interval(nanoseconds=2 * 1000),
        ),
        (
            api.interval(nanoseconds=2).to_unit('ns'),
            api.interval(nanoseconds=2),
        ),
    ],
)
def test_downconvert_second_parts(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (api.interval(hours=2).to_unit('h'), api.interval(hours=2)),
        (api.interval(hours=2).to_unit('m'), api.interval(minutes=2 * 60)),
        (api.interval(hours=2).to_unit('s'), api.interval(seconds=2 * 3600)),
        (
            api.interval(hours=2).to_unit('ms'),
            api.interval(milliseconds=2 * 3600000),
        ),
        (
            api.interval(hours=2).to_unit('us'),
            api.interval(microseconds=2 * 3600000000),
        ),
        (
            api.interval(hours=2).to_unit('ns'),
            api.interval(nanoseconds=2 * 3600000000000),
        ),
    ],
)
def test_downconvert_hours(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (api.interval(weeks=2).to_unit('D'), api.interval(days=2 * 7)),
        (api.interval(weeks=2).to_unit('h'), api.interval(hours=2 * 7 * 24)),
        (api.interval(days=2).to_unit('D'), api.interval(days=2)),
        (api.interval(days=2).to_unit('h'), api.interval(hours=2 * 24)),
        (api.interval(days=2).to_unit('m'), api.interval(minutes=2 * 1440)),
        (api.interval(days=2).to_unit('s'), api.interval(seconds=2 * 86400)),
        (
            api.interval(days=2).to_unit('ms'),
            api.interval(milliseconds=2 * 86400000),
        ),
        (
            api.interval(days=2).to_unit('us'),
            api.interval(microseconds=2 * 86400000000),
        ),
        (
            api.interval(days=2).to_unit('ns'),
            api.interval(nanoseconds=2 * 86400000000000),
        ),
    ],
)
def test_downconvert_day(case, expected):
    assert isinstance(case, ir.IntervalScalar)
    assert isinstance(expected, ir.IntervalScalar)
    assert case.type().unit == expected.type().unit


@pytest.mark.parametrize(
    ('a', 'b', 'unit'),
    [
        (api.interval(days=1), api.interval(days=3), 'D'),
        (api.interval(seconds=1), api.interval(hours=10), 's'),
        (api.interval(hours=3), api.interval(days=2), 'h'),
    ],
)
def test_combine_with_different_kinds(a, b, unit):
    assert (a + b).type().unit == unit


@pytest.mark.parametrize(
    ('case', 'expected'),
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


def test_interval_time_expr(table):
    c = table.k
    x = api.interval(hours=1)

    expr = x + c
    assert isinstance(expr, ir.TimeColumn)
    assert isinstance(expr.op(), ops.TimeAdd)

    # test radd
    expr = c + x
    assert isinstance(expr, ir.TimeColumn)
    assert isinstance(expr.op(), ops.TimeAdd)

    expr = x - c
    assert isinstance(expr, ir.TimeColumn)
    assert isinstance(expr.op(), ops.TimeSub)

    # test radd
    expr = c - x
    assert isinstance(expr, ir.TimeColumn)
    assert isinstance(expr.op(), ops.TimeSub)


def test_interval_date_expr(table):
    c = table.j
    x = api.interval(days=1)

    expr = x + c
    assert isinstance(expr, ir.DateColumn)
    assert isinstance(expr.op(), ops.DateAdd)

    # test radd
    expr = c + x
    assert isinstance(expr, ir.DateColumn)
    assert isinstance(expr.op(), ops.DateAdd)

    expr = x - c
    assert isinstance(expr, ir.DateColumn)
    assert isinstance(expr.op(), ops.DateSub)

    # test radd
    expr = c - x
    assert isinstance(expr, ir.DateColumn)
    assert isinstance(expr.op(), ops.DateSub)


def test_interval_timestamp_expr(table):
    c = table.i
    x = api.interval(seconds=1)

    expr = x + c
    assert isinstance(expr, ir.TimestampColumn)
    assert isinstance(expr.op(), ops.TimestampAdd)

    # test radd
    expr = c + x
    assert isinstance(expr, ir.TimestampColumn)
    assert isinstance(expr.op(), ops.TimestampAdd)

    expr = x - c
    assert isinstance(expr, ir.TimestampColumn)
    assert isinstance(expr.op(), ops.TimestampSub)

    # test radd
    expr = c - x
    assert isinstance(expr, ir.TimestampColumn)
    assert isinstance(expr.op(), ops.TimestampSub)


@pytest.mark.xfail(raises=AssertionError, reason='NYI')
def test_compound_offset():
    # These are not yet allowed (e.g. 1 month + 1 hour)
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_offset_months():
    assert False


@pytest.mark.parametrize(
    'literal',
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


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (api.interval(weeks=3), "Literal[interval<int8>(unit='W')]\n  3"),
        (api.interval(months=3), "Literal[interval<int8>(unit='M')]\n  3"),
        (
            api.interval(seconds=-10),
            "Literal[interval<int8>(unit='s')]\n  -10",
        ),
    ],
)
def test_interval_repr(expr, expected):
    assert repr(expr) == expected


def test_timestamp_arithmetics():
    ts1 = api.timestamp(datetime.datetime.now())
    ts2 = api.timestamp(datetime.datetime.today())

    i1 = api.interval(minutes=30)

    # TODO: raise for unsupported operations too
    for expr in [ts2 - ts1, ts1 - ts2]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimestampDiff)
        assert expr.type() == dt.Interval('s', dt.int32)

    for expr in [ts1 - i1, ts2 - i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampSub)

    for expr in [ts1 + i1, ts2 + i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampAdd)


def test_date_arithmetics():
    d1 = api.date('2015-01-02')
    d2 = api.date('2017-01-01')
    i1 = api.interval(weeks=3)

    for expr in [d1 - d2, d2 - d1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.DateDiff)
        assert expr.type() == dt.Interval('D', dt.int32)

    for expr in [d1 - i1, d2 - i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateSub)

    for expr in [d1 + i1, d2 + i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateAdd)


def test_time_arithmetics():
    t1 = api.time('18:00')
    t2 = api.time('19:12')
    i1 = api.interval(minutes=3)

    for expr in [t1 - t2, t2 - t1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimeDiff)
        assert expr.type() == dt.Interval('s', dt.int32)

    for expr in [t1 - i1, t2 - i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeSub)

    for expr in [t1 + i1, t2 + i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeAdd)


def test_invalid_date_arithmetics():
    d1 = api.date('2015-01-02')
    i1 = api.interval(seconds=300)
    i2 = api.interval(minutes=15)
    i3 = api.interval(hours=1)

    for i in [i1, i2, i3]:
        with pytest.raises(TypeError):
            d1 - i
        with pytest.raises(TypeError):
            d1 + i


@pytest.mark.parametrize(
    ('prop', 'expected_unit'),
    [
        ('nanoseconds', 'ns'),
        ('microseconds', 'us'),
        ('milliseconds', 'ms'),
        ('seconds', 's'),
        ('minutes', 'm'),
        ('hours', 'h'),
        ('days', 'D'),
        ('weeks', 'W'),
    ],
)
def test_interval_properties(prop, expected_unit):
    i = api.interval(seconds=3600)
    assert getattr(i, prop).type().unit == expected_unit


@pytest.mark.parametrize(
    'interval',
    [api.interval(years=1), api.interval(quarters=4), api.interval(months=12)],
)
@pytest.mark.parametrize(
    ('prop', 'expected_unit'),
    [('months', 'M'), ('quarters', 'Q'), ('years', 'Y')],
)
def test_interval_monthly_properties(interval, prop, expected_unit):
    assert getattr(interval, prop).type().unit == expected_unit


@pytest.mark.parametrize(
    ('interval', 'prop'),
    [
        (api.interval(hours=48), 'months'),
        (api.interval(years=2), 'seconds'),
        (api.interval(quarters=1), 'weeks'),
    ],
)
def test_unsupported_properties(interval, prop):
    with pytest.raises(ValueError):
        getattr(interval, prop)


@pytest.mark.parametrize('column', ['a', 'b', 'c', 'd'])  # integer columns
@pytest.mark.parametrize(
    'unit', ['Y', 'Q', 'M', 'D', 'W', 'h', 'm', 's', 'ms', 'us', 'ns']
)
def test_integer_to_interval(column, unit, table):
    c = table[column]
    i = c.to_interval(unit)
    assert isinstance(i, ir.IntervalColumn)
    assert i.type().value_type == c.type()
    assert i.type().unit == unit


@pytest.mark.parametrize(
    'unit', ['Y', 'Q', 'M', 'D', 'W', 'h', 'm', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize(
    'operands',
    [
        lambda t, u: (api.interval(3, unit=u), api.interval(2, unit=u)),
        lambda t, u: (api.interval(3, unit=u), api.interval(3, unit=u)),
        lambda t, u: (t.c.to_interval(unit=u), api.interval(2, unit=u)),
        lambda t, u: (t.c.to_interval(unit=u), t.d.to_interval(unit=u)),
    ],
)
@pytest.mark.parametrize(
    'operator',
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
    'operands',
    [
        lambda t: (api.date('2016-01-01'), api.date('2016-02-02')),
        lambda t: (t.j, api.date('2016-01-01')),
        lambda t: (api.date('2016-01-01'), t.j),
        lambda t: (t.j, t.i.date()),
        lambda t: (t.i.date(), t.j),
    ],
    ids=[
        'literal-literal',
        'column-literal',
        'literal-column',
        'column-casted',
        'casted-column',
    ],
)
@pytest.mark.parametrize(
    'interval',
    [
        lambda t: api.interval(years=4),
        lambda t: api.interval(quarters=4),
        lambda t: api.interval(months=3),
        lambda t: api.interval(weeks=2),
        lambda t: api.interval(days=1),
        lambda t: t.c.to_interval(unit='Y'),
        lambda t: t.c.to_interval(unit='M'),
        lambda t: t.c.to_interval(unit='W'),
        lambda t: t.c.to_interval(unit='D'),
    ],
    ids=[
        'years',
        'quarters',
        'months',
        'weeks',
        'days',
        'to-years',
        'to-months',
        'to-weeks',
        'to-days',
    ],
)
@pytest.mark.parametrize(
    'arithmetic',
    [lambda a, i: a - i, lambda a, i: a + i, lambda a, i: i + a],
    ids=['subtract', 'radd', 'add'],
)
@pytest.mark.parametrize(
    'operator',
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
def test_complex_date_comparisons(
    operands, interval, arithmetic, operator, table
):
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
    expr = (c - c).name('foo')
    assert expr._name == 'foo'


@pytest.mark.parametrize(
    'operand',
    [lambda t: api.timestamp(datetime.datetime.now()), lambda t: t.i],
    ids=['column', 'literal'],
)
@pytest.mark.parametrize(
    'unit',
    [
        'Y',
        'y',
        'year',
        'YEAR',
        'YYYY',
        'SYYYY',
        'YYY',
        'YY',
        'Q',
        'q',
        'quarter',
        'QUARTER',
        'M',
        'month',
        'MONTH',
        'w',
        'W',
        'week',
        'WEEK',
        'd',
        'J',
        'day',
        'DAY',
        'h',
        'H',
        'HH24',
        'hour',
        'HOUR',
        'm',
        'MI',
        'minute',
        'MINUTE',
        's',
        'second',
        'SECOND',
        'ms',
        'millisecond',
        'MILLISECOND',
        'us',
        'microsecond',
        'MICROSECOND',
        'ns',
        'nanosecond',
        'NANOSECOND',
    ],
)
def test_timestamp_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.TimestampValue)
    assert isinstance(expr.op(), ops.TimestampTruncate)


@pytest.mark.parametrize(
    'operand', [lambda t: api.date('2018-01-01'), lambda t: t.j]
)
@pytest.mark.parametrize('unit', ['Y', 'Q', 'M', 'D', 'W'])
def test_date_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.DateValue)
    assert isinstance(expr.op(), ops.DateTruncate)


@pytest.mark.parametrize(
    'operand', [lambda t: api.time('18:00'), lambda t: t.k]
)
@pytest.mark.parametrize('unit', ['h', 'm', 's', 'ms', 'us', 'ns'])
def test_time_truncate(table, operand, unit):
    expr = operand(table).truncate(unit)
    assert isinstance(expr, ir.TimeValue)
    assert isinstance(expr.op(), ops.TimeTruncate)
