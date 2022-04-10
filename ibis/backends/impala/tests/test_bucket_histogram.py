import pytest

from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


BUCKETS = [0, 10, 25, 50]


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda f: f.bucket(BUCKETS),
            """\
CASE
  WHEN (0 <= `f`) AND (`f` < 10) THEN 0
  WHEN (10 <= `f`) AND (`f` < 25) THEN 1
  WHEN (25 <= `f`) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END""",
            id="default",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, close_extreme=False),
            """\
CASE
  WHEN (0 <= `f`) AND (`f` < 10) THEN 0
  WHEN (10 <= `f`) AND (`f` < 25) THEN 1
  WHEN (25 <= `f`) AND (`f` < 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END""",
            id="close_extreme_false",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, closed="right"),
            """\
CASE
  WHEN (0 <= `f`) AND (`f` <= 10) THEN 0
  WHEN (10 < `f`) AND (`f` <= 25) THEN 1
  WHEN (25 < `f`) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END""",
            id="closed_right",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, closed="right", close_extreme=False),
            """\
CASE
  WHEN (0 < `f`) AND (`f` <= 10) THEN 0
  WHEN (10 < `f`) AND (`f` <= 25) THEN 1
  WHEN (25 < `f`) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END""",
            id="close_extreme_false_closed_right",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, include_under=True),
            """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (0 <= `f`) AND (`f` < 10) THEN 1
  WHEN (10 <= `f`) AND (`f` < 25) THEN 2
  WHEN (25 <= `f`) AND (`f` <= 50) THEN 3
  ELSE CAST(NULL AS tinyint)
END""",
            id="include_under",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, include_under=True, include_over=True),
            """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (0 <= `f`) AND (`f` < 10) THEN 1
  WHEN (10 <= `f`) AND (`f` < 25) THEN 2
  WHEN (25 <= `f`) AND (`f` <= 50) THEN 3
  WHEN 50 < `f` THEN 4
  ELSE CAST(NULL AS tinyint)
END""",
            id="include_under_include_over",
        ),
        pytest.param(
            lambda f: f.bucket(
                BUCKETS,
                close_extreme=False,
                include_under=True,
                include_over=True,
            ),
            """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (0 <= `f`) AND (`f` < 10) THEN 1
  WHEN (10 <= `f`) AND (`f` < 25) THEN 2
  WHEN (25 <= `f`) AND (`f` < 50) THEN 3
  WHEN 50 <= `f` THEN 4
  ELSE CAST(NULL AS tinyint)
END""",
            id="close_extreme_false_include_under_include_over",
        ),
        pytest.param(
            lambda f: f.bucket(
                BUCKETS,
                closed="right",
                close_extreme=False,
                include_under=True,
            ),
            """\
CASE
  WHEN `f` <= 0 THEN 0
  WHEN (0 < `f`) AND (`f` <= 10) THEN 1
  WHEN (10 < `f`) AND (`f` <= 25) THEN 2
  WHEN (25 < `f`) AND (`f` <= 50) THEN 3
  ELSE CAST(NULL AS tinyint)
END""",
            id="closed_right_close_extreme_false_include_under",
        ),
        pytest.param(
            lambda f: f.bucket(
                [10],
                closed="right",
                include_over=True,
                include_under=True,
            ),
            """\
CASE
  WHEN `f` <= 10 THEN 0
  WHEN 10 < `f` THEN 1
  ELSE CAST(NULL AS tinyint)
END""",
            id="closed_right_include_over_include_under",
        ),
        pytest.param(
            lambda f: f.bucket(
                [10],
                include_over=True,
                include_under=True,
            ),
            """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN 10 <= `f` THEN 1
  ELSE CAST(NULL AS tinyint)
END""",
            id="include_over_include_under",
        ),
        # Because the bucket result is an integer, no explicit cast is
        # necessary
        pytest.param(
            lambda f: f.bucket(
                [10], include_over=True, include_under=True
            ).cast('int32'),
            """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN 10 <= `f` THEN 1
  ELSE CAST(NULL AS tinyint)
END""",
            id="include_over_include_under",
        ),
        pytest.param(
            lambda f: f.bucket(
                [10], include_over=True, include_under=True
            ).cast('double'),
            """\
CAST(CASE
  WHEN `f` < 10 THEN 0
  WHEN 10 <= `f` THEN 1
  ELSE CAST(NULL AS tinyint)
END AS double)""",
            id="include_over_include_under",
        ),
    ],
)
def test_bucket_to_case(table, expr_fn, expected):
    expr = expr_fn(table.f)
    result = translate(expr)
    assert result == expected


def test_bucket_assign_labels(table):
    buckets = [0, 10, 25, 50]
    bucket = table.f.bucket(buckets, include_under=True)

    size = table.group_by(bucket.name('tier')).size()
    labelled = size.tier.label(
        ['Under 0', '0 to 10', '10 to 25', '25 to 50'], nulls='error'
    ).name('tier2')
    expr = size[labelled, size['count']]

    expected = """\
SELECT
  CASE `tier`
    WHEN 0 THEN 'Under 0'
    WHEN 1 THEN '0 to 10'
    WHEN 2 THEN '10 to 25'
    WHEN 3 THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`, `count`
FROM (
  SELECT
    CASE
      WHEN `f` < 0 THEN 0
      WHEN (0 <= `f`) AND (`f` < 10) THEN 1
      WHEN (10 <= `f`) AND (`f` < 25) THEN 2
      WHEN (25 <= `f`) AND (`f` <= 50) THEN 3
      ELSE CAST(NULL AS tinyint)
    END AS `tier`, count(*) AS `count`
  FROM alltypes
  GROUP BY 1
) t0"""

    result = ImpalaCompiler.to_sql(expr)
    assert result == expected

    with pytest.raises(ValueError):
        size.tier.label(list("abc"))

    with pytest.raises(ValueError):
        size.tier.label(list("abcde"))
