import datetime

import pandas as pd
import pytest

import ibis
import ibis.backends.bigquery as bq
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr

pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_timestamp_accepts_date_literals(alltypes, project_id):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp).name('param_0')
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    result = expr.compile(params=params)
    expected = f"""\
SELECT *, @param AS `param`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected


@pytest.mark.parametrize(
    ('distinct', 'expected_keyword'), [(True, 'DISTINCT'), (False, 'ALL')]
)
def test_union(alltypes, distinct, expected_keyword, project_id):
    expr = alltypes.union(alltypes, distinct=distinct)
    result = expr.compile()
    expected = f"""\
SELECT *
FROM `{project_id}.testing.functional_alltypes`
UNION {expected_keyword}
SELECT *
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected


def test_ieee_divide(alltypes, project_id):
    expr = alltypes.double_col / 0
    result = expr.compile()
    expected = f"""\
SELECT IEEE_DIVIDE(`double_col`, 0) AS `tmp`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected


def test_identical_to(alltypes, project_id):
    t = alltypes
    pred = t.string_col.identical_to('a') & t.date_string_col.identical_to('b')
    expr = t[pred]
    result = expr.compile()
    expected = f"""\
SELECT *
FROM `{project_id}.testing.functional_alltypes`
WHERE (((`string_col` IS NULL) AND ('a' IS NULL)) OR (`string_col` = 'a')) AND
      (((`date_string_col` IS NULL) AND ('b' IS NULL)) OR (`date_string_col` = 'b'))"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize('timezone', [None, 'America/New_York'])
def test_to_timestamp(alltypes, timezone, project_id):
    expr = alltypes.date_string_col.to_timestamp('%F', timezone)
    result = expr.compile()
    if timezone:
        expected = f"""\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`, 'America/New_York') AS `tmp`
FROM `{project_id}.testing.functional_alltypes`"""
    else:
        expected = f"""\
SELECT PARSE_TIMESTAMP('%F', `date_string_col`) AS `tmp`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype'),
    [
        (datetime.date(2017, 1, 1), "DATE '2017-01-01'", dt.date),
        (pd.Timestamp('2017-01-01'), "DATE '2017-01-01'", dt.date,),
        ('2017-01-01', "DATE '2017-01-01'", dt.date),
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
    ],
)
def test_literal_date(case, expected, dtype):
    expr = ibis.literal(case, type=dtype).year()
    result = ibis.bigquery.compile(expr)
    assert result == f"SELECT EXTRACT(year from {expected}) AS `tmp`"


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype', 'strftime_func'),
    [
        (
            datetime.date(2017, 1, 1),
            "DATE '2017-01-01'",
            dt.date,
            'FORMAT_DATE',
        ),
        (
            pd.Timestamp('2017-01-01'),
            "DATE '2017-01-01'",
            dt.date,
            'FORMAT_DATE',
        ),
        ('2017-01-01', "DATE '2017-01-01'", dt.date, 'FORMAT_DATE',),
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
            'FORMAT_TIMESTAMP',
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
            'FORMAT_TIMESTAMP',
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
            'FORMAT_TIMESTAMP',
        ),
    ],
)
def test_day_of_week(case, expected, dtype, strftime_func):
    date_var = ibis.literal(case, type=dtype)
    expr_index = date_var.day_of_week.index()
    result = ibis.bigquery.compile(expr_index)
    assert (
        result
        == f"SELECT MOD(EXTRACT(DAYOFWEEK FROM {expected}) + 5, 7) AS `tmp`"
    )

    expr_name = date_var.day_of_week.full_name()
    result = ibis.bigquery.compile(expr_name)
    if strftime_func == 'FORMAT_TIMESTAMP':
        assert (
            result
            == f"SELECT {strftime_func}('%A', {expected}, 'UTC') AS `tmp`"
        )
    else:
        assert result == f"SELECT {strftime_func}('%A', {expected}) AS `tmp`"


@pytest.mark.parametrize(
    ("case", "expected", "dtype"),
    [
        ("test of hash", "'test of hash'", dt.string,),
        (b"test of hash", "FROM_BASE64('dGVzdCBvZiBoYXNo')", dt.binary,),
    ],
)
def test_hash(case, expected, dtype):
    string_var = ibis.literal(case, type=dtype)
    expr = string_var.hash(how="farm_fingerprint")
    result = ibis.bigquery.compile(expr)
    assert result == f"SELECT farm_fingerprint({expected}) AS `tmp`"


@pytest.mark.parametrize(
    ("case", "expected", "how", "dtype"),
    [
        ("test", "md5('test')", "md5", dt.string,),
        (b"test", "md5(FROM_BASE64('dGVzdA=='))", "md5", dt.binary,),
        ("test", "sha1('test')", "sha1", dt.string,),
        (b"test", "sha1(FROM_BASE64('dGVzdA=='))", "sha1", dt.binary,),
        ("test", "sha256('test')", "sha256", dt.string,),
        (b"test", "sha256(FROM_BASE64('dGVzdA=='))", "sha256", dt.binary,),
        ("test", "sha512('test')", "sha512", dt.string,),
        (b"test", "sha512(FROM_BASE64('dGVzdA=='))", "sha512", dt.binary,),
    ],
)
def test_hashbytes(case, expected, how, dtype):
    var = ibis.literal(case, type=dtype)
    expr = var.hashbytes(how=how)
    result = ibis.bigquery.compile(expr)
    assert result == f"SELECT {expected} AS `tmp`"


@pytest.mark.parametrize(
    ('case', 'expected', 'dtype'),
    [
        (
            datetime.datetime(2017, 1, 1, 4, 55, 59),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
        (
            '2017-01-01 04:55:59',
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
        (
            pd.Timestamp('2017-01-01 04:55:59'),
            "TIMESTAMP '2017-01-01 04:55:59'",
            dt.timestamp,
        ),
        (datetime.time(4, 55, 59), "TIME '04:55:59'", dt.time),
        ('04:55:59', "TIME '04:55:59'", dt.time),
    ],
)
def test_literal_timestamp_or_time(case, expected, dtype):
    expr = ibis.literal(case, type=dtype).hour()
    result = ibis.bigquery.compile(expr)
    assert result == f"SELECT EXTRACT(hour from {expected}) AS `tmp`"


def test_window_function(alltypes, project_id):
    t = alltypes
    w1 = ibis.window(
        preceding=1, following=0, group_by='year', order_by='timestamp_col'
    )
    expr = t.mutate(win_avg=t.float_col.mean().over(w1))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w2 = ibis.window(
        preceding=0, following=2, group_by='year', order_by='timestamp_col'
    )
    expr = t.mutate(win_avg=t.float_col.mean().over(w2))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS `win_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w3 = ibis.window(
        preceding=(4, 2), group_by='year', order_by='timestamp_col'
    )
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ROWS BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_range_window_function(alltypes, project_id):
    t = alltypes
    w = ibis.range_window(
        preceding=1, following=0, group_by='year', order_by='month'
    )
    expr = t.mutate(two_month_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `month` RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS `two_month_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    w3 = ibis.range_window(
        preceding=(4, 2), group_by='year', order_by='timestamp_col'
    )
    expr = t.mutate(win_avg=t.float_col.mean().over(w3))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    ('preceding', 'value'),
    [
        (5, 5),
        (ibis.interval(nanoseconds=1), 0.001),
        (ibis.interval(microseconds=1), 1),
        (ibis.interval(seconds=1), 1000000),
        (ibis.interval(minutes=1), 1000000 * 60),
        (ibis.interval(hours=1), 1000000 * 60 * 60),
        (ibis.interval(days=1), 1000000 * 60 * 60 * 24),
        (2 * ibis.interval(days=1), 1000000 * 60 * 60 * 24 * 2),
        (ibis.interval(weeks=1), 1000000 * 60 * 60 * 24 * 7),
    ],
)
def test_trailing_range_window(alltypes, preceding, value, project_id):
    t = alltypes
    w = ibis.trailing_range_window(
        preceding=preceding, order_by=t.timestamp_col
    )
    expr = t.mutate(win_avg=t.float_col.mean().over(w))
    result = expr.compile()
    expected = f"""\
SELECT *,
       avg(`float_col`) OVER (ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN {value} PRECEDING AND CURRENT ROW) AS `win_avg`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


@pytest.mark.parametrize(
    ('preceding', 'value'), [(ibis.interval(years=1), None)]
)
def test_trailing_range_window_unsupported(alltypes, preceding, value):
    t = alltypes
    w = ibis.trailing_range_window(
        preceding=preceding, order_by=t.timestamp_col
    )
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
    ],
)
def test_union_cte(
    alltypes, distinct1, distinct2, expected1, expected2, project_id
):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2, distinct=distinct1).union(
        expr3, distinct=distinct2
    )
    result = expr.compile()
    expected = f"""\
WITH t0 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM `{project_id}.testing.functional_alltypes`
  GROUP BY 1
)
SELECT *
FROM t0
{expected1}
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM `{project_id}.testing.functional_alltypes`
GROUP BY 1
{expected2}
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM `{project_id}.testing.functional_alltypes`
GROUP BY 1"""
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


def test_bool_reducers(alltypes, project_id):
    b = alltypes.bool_col
    expr = b.mean()
    result = expr.compile()
    expected = f"""\
SELECT avg(CAST(`bool_col` AS INT64)) AS `mean`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected

    expr2 = b.sum()
    result = expr2.compile()
    expected = f"""\
SELECT sum(CAST(`bool_col` AS INT64)) AS `sum`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected


def test_bool_reducers_where(alltypes, project_id):
    b = alltypes.bool_col
    m = alltypes.month
    expr = b.mean(where=m > 6)
    result = expr.compile()
    expected = f"""\
SELECT avg(CASE WHEN `month` > 6 THEN CAST(`bool_col` AS INT64) ELSE NULL END) AS `mean`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected

    expr2 = b.sum(where=((m > 6) & (m < 10)))
    result = expr2.compile()
    expected = f"""\
SELECT sum(CASE WHEN (`month` > 6) AND (`month` < 10) THEN CAST(`bool_col` AS INT64) ELSE NULL END) AS `sum`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_approx_nunique(alltypes, project_id):
    d = alltypes.double_col
    expr = d.approx_nunique()
    result = expr.compile()
    expected = f"""\
SELECT APPROX_COUNT_DISTINCT(`double_col`) AS `approx_nunique`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected

    b = alltypes.bool_col
    m = alltypes.month
    expr2 = b.approx_nunique(where=m > 6)
    result = expr2.compile()
    expected = f"""\
SELECT APPROX_COUNT_DISTINCT(CASE WHEN `month` > 6 THEN `bool_col` ELSE NULL END) AS `approx_nunique`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_approx_median(alltypes, project_id):
    d = alltypes.double_col
    expr = d.approx_median()
    result = expr.compile()
    expected = f"""\
SELECT APPROX_QUANTILES(`double_col`, 2)[OFFSET(1)] AS `approx_median`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected

    m = alltypes.month
    expr2 = d.approx_median(where=m > 6)
    result = expr2.compile()
    expected = f"""\
SELECT APPROX_QUANTILES(CASE WHEN `month` > 6 THEN `double_col` ELSE NULL END, 2)[OFFSET(1)] AS `approx_median`
FROM `{project_id}.testing.functional_alltypes`"""  # noqa: E501
    assert result == expected


def test_cov(alltypes, project_id):
    d = alltypes.double_col
    expr = d.cov(d)
    result = expr.compile()
    expected = f"""\
SELECT COVAR_SAMP(`double_col`, `double_col`) AS `tmp`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected

    expr = d.cov(d, how='pop')
    result = expr.compile()
    expected = f"""\
SELECT COVAR_POP(`double_col`, `double_col`) AS `tmp`
FROM `{project_id}.testing.functional_alltypes`"""
    assert result == expected

    with pytest.raises(ValueError):
        d.cov(d, how='error')


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
    ],
)
def test_temporal_truncate(unit, expected_unit, expected_func):
    t = ibis.table([('a', getattr(dt, expected_func.lower()))], name='t')
    expr = t.a.truncate(unit)
    result = ibis.bigquery.compile(expr)
    expected = f"""\
SELECT {expected_func}_TRUNC(`a`, {expected_unit}) AS `tmp`
FROM t"""
    assert result == expected


@pytest.mark.parametrize('kind', ['date', 'time'])
def test_extract_temporal_from_timestamp(kind):
    t = ibis.table([('ts', dt.timestamp)], name='t')
    expr = getattr(t.ts, kind)()
    result = ibis.bigquery.compile(expr)
    expected = f"""\
SELECT {kind.upper()}(`ts`) AS `tmp`
FROM t"""
    assert result == expected


def test_now():
    expr = ibis.now()
    result = ibis.bigquery.compile(expr)
    expected = 'SELECT CURRENT_TIMESTAMP() AS `tmp`'
    assert result == expected


def test_bucket():
    t = ibis.table([('value', 'double')], name='t')
    buckets = [0, 1, 3]
    expr = t.value.bucket(buckets).name('foo')
    result = ibis.bigquery.compile(expr)
    expected = """\
SELECT
  CASE
    WHEN (`value` >= 0) AND (`value` < 1) THEN 0
    WHEN (`value` >= 1) AND (`value` <= 3) THEN 1
    ELSE CAST(NULL AS INT64)
  END AS `tmp`
FROM t"""
    assert result == expected


@pytest.mark.parametrize(
    ('kind', 'begin', 'end', 'expected'),
    [
        ('preceding', None, 1, 'UNBOUNDED PRECEDING AND 1 PRECEDING'),
        ('following', 1, None, '1 FOLLOWING AND UNBOUNDED FOLLOWING'),
    ],
)
def test_window_unbounded(kind, begin, end, expected):
    t = ibis.table([('a', 'int64')], name='t')
    kwargs = {kind: (begin, end)}
    expr = t.a.sum().over(ibis.window(**kwargs))
    result = ibis.bigquery.compile(expr)
    assert (
        result
        == f"""\
SELECT sum(`a`) OVER (ROWS BETWEEN {expected}) AS `tmp`
FROM t"""
    )


def test_large_compile():
    """
    Tests that compiling a large expression tree finishes
    within a reasonable amount of time
    """
    num_columns = 20
    num_joins = 7

    class MockBigQueryClient(bq.BigQueryClient):
        def __init__(self):
            pass

    names = [f"col_{i}" for i in range(num_columns)]
    schema = ibis.Schema(names, ['string'] * num_columns)
    ibis_client = MockBigQueryClient()
    table = TableExpr(
        ops.SQLQueryResult("select * from t", schema, ibis_client)
    )
    for _ in range(num_joins):
        table = table.mutate(dummy=ibis.literal(""))
        table = table.left_join(table, ["dummy"])[[table]]

    start = datetime.datetime.now()
    table.compile()
    delta = datetime.datetime.now() - start
    assert delta.total_seconds() < 10
