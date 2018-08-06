import datetime

import pytest

import pandas as pd

import ibis
import ibis.expr.datatypes as dt

from ibis.compat import PY2


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_timestamp_accepts_date_literals(alltypes, project_id):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp).name('param_0')
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    result = expr.compile(params=params)
    expected = """\
SELECT *, @param AS `param`
FROM `{}.testing.functional_alltypes`""".format(project_id)
    assert result == expected


@pytest.mark.parametrize(
    ('distinct', 'expected_keyword'),
    [
        (True, 'DISTINCT'),
        (False, 'ALL'),
    ]
)
def test_union(alltypes, distinct, expected_keyword, project_id):
    expr = alltypes.union(alltypes, distinct=distinct)
    result = expr.compile()
    expected = """\
SELECT *
FROM `{project}.testing.functional_alltypes`
UNION {}
SELECT *
FROM `{project}.testing.functional_alltypes`""".format(
        expected_keyword, project=project_id)
    assert result == expected


def test_ieee_divide(alltypes, project_id):
    expr = alltypes.double_col / 0
    result = expr.compile()
    expected = """\
SELECT IEEE_DIVIDE(`double_col`, 0) AS `tmp`
FROM `{}.testing.functional_alltypes`""".format(project_id)
    assert result == expected


def test_identical_to(alltypes, project_id):
    t = alltypes
    pred = t.string_col.identical_to('a') & t.date_string_col.identical_to('b')
    expr = t[pred]
    result = expr.compile()
    expected = """\
SELECT *
FROM `{}.testing.functional_alltypes`
WHERE (((`string_col` IS NULL) AND ('a' IS NULL)) OR (`string_col` = 'a')) AND
      (((`date_string_col` IS NULL) AND ('b' IS NULL)) OR (`date_string_col` = 'b'))""".format(  # noqa: E501
        project_id)
    assert result == expected


@pytest.mark.parametrize(
    'timezone',
    [
        None,
        'America/New_York'
    ]
)
def test_to_timestamp(alltypes, timezone, project_id):
    expr = alltypes.date_string_col.to_timestamp('%F', timezone)
    result = expr.compile()
    if timezone:
        expected = """\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`, 'America/New_York') AS `tmp`
FROM `{}.testing.functional_alltypes`""".format(project_id)
    else:
        expected = """\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`) AS `tmp`
FROM `{}.testing.functional_alltypes`""".format(project_id)
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


def test_window_function(alltypes, project_id):
    t = alltypes
    w1 = ibis.window(preceding=1, following=0,
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w1))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `{}.testing.functional_alltypes`""".format(project_id)  # noqa: E501
    assert result == expected

    w2 = ibis.window(preceding=0, following=2,
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w2))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS `win_avg`
FROM `{}.testing.functional_alltypes`""".format(project_id)  # noqa: E501
    assert result == expected

    w3 = ibis.window(preceding=(4, 2),
                     group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `{}.testing.functional_alltypes`""".format(project_id)  # noqa: E501
    assert result == expected


def test_range_window_function(alltypes, project_id):
    t = alltypes
    w = ibis.range_window(preceding=1, following=0,
                          group_by='year', order_by='month')
    expr = t.mutate(two_month_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `month` RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS `two_month_avg`
FROM `{}.testing.functional_alltypes`""".format(project_id)  # noqa: E501
    assert result == expected

    w3 = ibis.range_window(preceding=(4, 2),
                           group_by='year', order_by='timestamp_col')
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `{}.testing.functional_alltypes`""".format(project_id)  # noqa: E501
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
def test_trailing_range_window(alltypes, preceding, value, project_id):
    t = alltypes
    w = ibis.trailing_range_window(preceding=preceding,
                                   order_by=t.timestamp_col)
    expr = t.mutate(win_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = """\
SELECT *,
       avg(`float_col`) OVER (ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN {} PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `{}.testing.functional_alltypes`""".format(  # noqa: E501
        value, project_id)
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


@pytest.mark.parametrize(
    ('distinct1', 'distinct2', 'expected1', 'expected2'),
    [
        (True, True, 'UNION DISTINCT', 'UNION DISTINCT'),
        (True, False, 'UNION DISTINCT', 'UNION ALL'),
        (False, True, 'UNION ALL', 'UNION DISTINCT'),
        (False, False, 'UNION ALL', 'UNION ALL'),
    ]
)
def test_union_cte(
        alltypes, distinct1, distinct2, expected1, expected2, project_id):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(
        expr2, distinct=distinct1).union(expr3, distinct=distinct2)
    result = expr.compile()
    expected = """\
WITH t0 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM `{project}.testing.functional_alltypes`
  GROUP BY 1
)
SELECT *
FROM t0
{}
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM `{project}.testing.functional_alltypes`
GROUP BY 1
{}
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM `{project}.testing.functional_alltypes`
GROUP BY 1""".format(expected1, expected2, project=project_id)
    assert result == expected


def test_projection_fusion_only_peeks_at_immediate_parent():
    schema = [
        ('file_date', 'timestamp'),
        ('PARTITIONTIME', 'date'),
        ('val', 'int64'),
    ]
    table = ibis.table(schema, name='unbound_table')
    table = table[table.PARTITIONTIME < ibis.date('2017-01-01')]
    table = table.mutate(file_date=table.file_date.cast('date'))
    table = table[table.file_date < ibis.date('2017-01-01')]
    table = table.mutate(XYZ=table.val * 2)
    expr = table.join(table.view())[table]
    result = ibis.bigquery.compile(expr)
    expected = """\
WITH t0 AS (
  SELECT *
  FROM unbound_table
  WHERE `PARTITIONTIME` < DATE '2017-01-01'
),
t1 AS (
  SELECT CAST(`file_date` AS DATE) AS `file_date`, `PARTITIONTIME`, `val`
  FROM t0
),
t2 AS (
  SELECT t1.*
  FROM t1
  WHERE t1.`file_date` < DATE '2017-01-01'
),
t3 AS (
  SELECT *, `val` * 2 AS `XYZ`
  FROM t2
)
SELECT t3.*
FROM t3
  CROSS JOIN t3 t4"""
    assert result == expected


def test_bool_reducers(alltypes):
    b = alltypes.bool_col
    expr = b.mean()
    result = expr.compile()
    expected = """\
SELECT avg(CAST(`bool_col` AS INT64)) AS `mean`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected

    expr2 = b.sum()
    result = expr2.compile()
    expected = """\
SELECT sum(CAST(`bool_col` AS INT64)) AS `sum`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected


def test_bool_reducers_where(alltypes):
    b = alltypes.bool_col
    m = alltypes.month
    expr = b.mean(where=m > 6)
    result = expr.compile()
    expected = """\
SELECT avg(CASE WHEN `month` > 6 THEN CAST(`bool_col` AS INT64) ELSE NULL END) AS `mean`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    expr2 = b.sum(where=((m > 6) & (m < 10)))
    result = expr2.compile()
    expected = """\
SELECT sum(CASE WHEN (`month` > 6) AND (`month` < 10) THEN CAST(`bool_col` AS INT64) ELSE NULL END) AS `sum`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_approx_nunique(alltypes):
    d = alltypes.double_col
    expr = d.approx_nunique()
    result = expr.compile()
    expected = """\
SELECT APPROX_COUNT_DISTINCT(`double_col`) AS `approx_nunique`
FROM `ibis-gbq.testing.functional_alltypes`"""
    assert result == expected

    b = alltypes.bool_col
    m = alltypes.month
    expr2 = b.approx_nunique(where=m > 6)
    result = expr2.compile()
    expected = """\
SELECT APPROX_COUNT_DISTINCT(CASE WHEN `month` > 6 THEN `bool_col` ELSE NULL END) AS `approx_nunique`
FROM `ibis-gbq.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    ('unit', 'expected_unit', 'expected_func'),
    [
        ('Y', 'YEAR', 'TIMESTAMP'),
        ('Q', 'QUARTER', 'TIMESTAMP'),
        ('M', 'MONTH', 'TIMESTAMP'),
        ('W', 'WEEK', 'TIMESTAMP'),
        ('D', 'DAY', 'TIMESTAMP'),
        ('h', 'HOUR', 'TIMESTAMP'),
        ('m', 'MINUTE', 'TIMESTAMP'),
        ('s', 'SECOND', 'TIMESTAMP'),
        ('ms', 'MILLISECOND', 'TIMESTAMP'),
        ('us', 'MICROSECOND', 'TIMESTAMP'),

        ('Y', 'YEAR', 'DATE'),
        ('Q', 'QUARTER', 'DATE'),
        ('M', 'MONTH', 'DATE'),
        ('W', 'WEEK', 'DATE'),
        ('D', 'DAY', 'DATE'),

        ('h', 'HOUR', 'TIME'),
        ('m', 'MINUTE', 'TIME'),
        ('s', 'SECOND', 'TIME'),
        ('ms', 'MILLISECOND', 'TIME'),
        ('us', 'MICROSECOND', 'TIME'),
    ]
)
def test_temporal_truncate(unit, expected_unit, expected_func):
    t = ibis.table([('a', getattr(dt, expected_func.lower()))], name='t')
    expr = t.a.truncate(unit)
    result = ibis.bigquery.compile(expr)
    expected = """\
SELECT {}_TRUNC(`a`, {}) AS `tmp`
FROM t""".format(expected_func, expected_unit)
    assert result == expected


@pytest.mark.parametrize(
    'kind',
    [
        'date',
        pytest.param(
            'time',
            marks=pytest.mark.xfail(
                PY2,
                reason='Time operations are not supported in Python 2',
                raises=ValueError
            )
        ),
    ]
)
def test_extract_temporal_from_timestamp(kind):
    t = ibis.table([('ts', dt.timestamp)], name='t')
    expr = getattr(t.ts, kind)()
    result = ibis.bigquery.compile(expr)
    expected = """\
SELECT {}(`ts`) AS `tmp`
FROM t""".format(kind.upper())
    assert result == expected


def test_now():
    expr = ibis.now()
    result = ibis.bigquery.compile(expr)
    expected = 'SELECT CURRENT_TIMESTAMP() AS `tmp`'
    assert result == expected
