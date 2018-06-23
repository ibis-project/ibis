import datetime

import pytest

import pandas as pd

import ibis
import ibis.expr.datatypes as dt


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_timestamp_accepts_date_literals(alltypes):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp).name('param_0')
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    result = expr.compile(params=params)
    expected = """\
SELECT *, @param AS `param`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected


@pytest.mark.parametrize(
    ('distinct', 'expected_keyword'),
    [
        (True, 'DISTINCT'),
        (False, 'ALL'),
    ]
)
def test_union(alltypes, distinct, expected_keyword):
    expr = alltypes.union(alltypes, distinct=distinct)
    result = expr.compile()
    expected = """\
SELECT *
FROM `ibis-gbq.testing.functional_alltypes`
UNION {}
SELECT *
FROM `ibis-gbq.testing.functional_alltypes`""".format(expected_keyword)
    assert result == expected


def test_ieee_divide(alltypes):
    expr = alltypes.double_col / 0
    result = expr.compile()
    expected = """\
SELECT IEEE_DIVIDE(`double_col`, 0) AS `tmp`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected


def test_identical_to(alltypes):
    t = alltypes
    pred = t.string_col.identical_to('a') & t.date_string_col.identical_to('b')
    expr = t[pred]
    result = expr.compile()
    expected = """\
SELECT *
FROM `ibis-gbq.testing.functional_alltypes`
WHERE (((`string_col` IS NULL) AND ('a' IS NULL)) OR (`string_col` = 'a')) AND
      (((`date_string_col` IS NULL) AND ('b' IS NULL)) OR (`date_string_col` = 'b'))"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    'timezone',
    [
        None,
        'America/New_York'
    ]
)
def test_to_timestamp(alltypes, timezone):
    expr = alltypes.date_string_col.to_timestamp('%F', timezone)
    result = expr.compile()
    if timezone:
        expected = """\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`, 'America/New_York') AS `tmp`
FROM `ibis-gbq.testing.functional_alltypes`"""
    else:
        expected = """\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`) AS `tmp`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype'),
    [
        (datetime.date(2017, 1, 1), "DATE '{}'".format('2017-01-01'), dt.date),
        (
            pd.Timestamp('2017-01-01'),
            "DATE '{}'".format('2017-01-01'),
            dt.date
        ),
        ('2017-01-01', "DATE '{}'".format('2017-01-01'), dt.date),
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
    ]
)
def test_literal_date(case, expected, dtype):
    expr = ibis.literal(case, type=dtype).year()
    result = ibis.bigquery.compile(expr)
    assert result == "SELECT EXTRACT(year from {}) AS `tmp`".format(expected)


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype', 'strftime_func'),
    [
        (
            datetime.date(2017, 1, 1),
            "DATE '{}'".format('2017-01-01'),
            dt.date,
            'FORMAT_DATE'
        ),
        (
            pd.Timestamp('2017-01-01'),
            "DATE '{}'".format('2017-01-01'),
            dt.date,
            'FORMAT_DATE'
        ),
        (
            '2017-01-01',
            "DATE '{}'".format('2017-01-01'),
            dt.date,
            'FORMAT_DATE'
        ),
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
            'FORMAT_TIMESTAMP'
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
            'FORMAT_TIMESTAMP'
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
            'FORMAT_TIMESTAMP'
        ),
    ]
)
def test_day_of_week(case, expected, dtype, strftime_func):
    date_var = ibis.literal(case, type=dtype)
    expr_index = date_var.day_of_week.index()
    result = ibis.bigquery.compile(expr_index)
    assert result == "SELECT MOD(EXTRACT(DAYOFWEEK FROM {}) + 5, 7) AS `tmp`".format(expected)  # noqa: E501

    expr_name = date_var.day_of_week.full_name()
    result = ibis.bigquery.compile(expr_name)
    if strftime_func == 'FORMAT_TIMESTAMP':
        assert result == "SELECT {}('%A', {}, 'UTC') AS `tmp`".format(
            strftime_func, expected
        )
    else:
        assert result == "SELECT {}('%A', {}) AS `tmp`".format(
            strftime_func, expected
        )


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype'),
    [
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '{}'".format('2017-01-01 04:55:59'),
            dt.timestamp,
        ),
        (
            datetime.time(4, 55, 59),
            "TIME '{}'".format('04:55:59'),
            dt.time,
        ),
        ('04:55:59', "TIME '{}'".format('04:55:59'), dt.time),
    ]
)
def test_literal_timestamp_or_time(case, expected, dtype):
    expr = ibis.literal(case, type=dtype).hour()
    result = ibis.bigquery.compile(expr)
    assert result == "SELECT EXTRACT(hour from {}) AS `tmp`".format(expected)


def test_window_function(alltypes):
    t = alltypes
    w1 = ibis.window(preceding=1, following=0,
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w1))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w2 = ibis.window(preceding=0, following=2,
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w2))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS `win_avg`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w3 = ibis.window(preceding=(4, 2),
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_range_window_function(alltypes):
    t = alltypes
    w = ibis.range_window(preceding=1, following=0,
                          group_by='year', order_by='month')
    expr = t.mutate(two_month_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `month` RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS `two_month_avg`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w3 = ibis.range_window(preceding=(4, 2),
                           group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    ('preceding', 'value'),
    [
        (5, 5),
        (ibis.nanosecond(), 0.001),
        (ibis.microsecond(), 1),
        (ibis.second(), 1000000),
        (ibis.minute(), 1000000 * 60),
        (ibis.hour(), 1000000 * 60 * 60),
        (ibis.day(), 1000000 * 60 * 60 * 24),
        (2 * ibis.day(), 1000000 * 60 * 60 * 24 * 2),
        (ibis.week(), 1000000 * 60 * 60 * 24 * 7),
    ]
)
def test_trailing_range_window(alltypes, preceding, value):
    t = alltypes
    w = ibis.trailing_range_window(preceding=preceding,
                                   order_by=t.timestamp_col)
    expr = t.mutate(win_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN {} PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `ibis-gbq.testing.functional_alltypes`""".format(value)  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    ('preceding', 'value'),
    [
        (ibis.year(), None),
    ]
)
def test_trailing_range_window_unsupported(alltypes, preceding, value):
    t = alltypes
    w = ibis.trailing_range_window(preceding=preceding,
                                   order_by=t.timestamp_col)
    expr = t.mutate(win_avg=t.float_col.mean().over(w))
    with pytest.raises(ValueError):
        expr.compile()


def test_union_cte(alltypes):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2).union(expr3)
    result = expr.compile()
    expected = """\
WITH t0 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM `ibis-gbq.testing.functional_alltypes`
  GROUP BY 1
),
t1 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM `ibis-gbq.testing.functional_alltypes`
  GROUP BY 1
)
SELECT *
FROM t0
UNION ALL
SELECT *
FROM t1
UNION ALL
SELECT *
FROM t1"""
    assert result == expected
