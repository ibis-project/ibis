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
