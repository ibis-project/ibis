from __future__ import annotations

import pytest

from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


BUCKETS = [0, 10, 25, 50]


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda f: f.bucket(BUCKETS), id="default"),
        pytest.param(
            lambda f: f.bucket(BUCKETS, close_extreme=False), id="close_extreme_false"
        ),
        pytest.param(lambda f: f.bucket(BUCKETS, closed="right"), id="closed_right"),
        pytest.param(
            lambda f: f.bucket(BUCKETS, closed="right", close_extreme=False),
            id="close_extreme_false_closed_right",
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, include_under=True), id="include_under"
        ),
        pytest.param(
            lambda f: f.bucket(BUCKETS, include_under=True, include_over=True),
            id="include_under_include_over",
        ),
        pytest.param(
            lambda f: f.bucket(
                BUCKETS, close_extreme=False, include_under=True, include_over=True
            ),
            id="close_extreme_false_include_under_include_over",
        ),
        pytest.param(
            lambda f: f.bucket(
                BUCKETS, closed="right", close_extreme=False, include_under=True
            ),
            id="closed_right_close_extreme_false_include_under",
        ),
        pytest.param(
            lambda f: f.bucket(
                [10], closed="right", include_over=True, include_under=True
            ),
            id="closed_right_include_over_include_under",
        ),
        pytest.param(
            lambda f: f.bucket([10], include_over=True, include_under=True),
            id="include_over_include_under",
        ),
        # Because the bucket result is an integer, no explicit cast is
        # necessary
        pytest.param(
            lambda f: f.bucket([10], include_over=True, include_under=True).cast(
                "int32"
            ),
            id="include_over_include_under",
        ),
        pytest.param(
            lambda f: f.bucket([10], include_over=True, include_under=True).cast(
                "double"
            ),
            id="include_over_include_under",
        ),
    ],
)
def test_bucket_to_case(table, expr_fn, snapshot):
    expr = expr_fn(table.f)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_bucket_assign_labels(table, snapshot):
    buckets = [0, 10, 25, 50]
    bucket = table.f.bucket(buckets, include_under=True)

    size = table.group_by(bucket.name("tier")).size()
    labelled = size.tier.label(
        ["Under 0", "0 to 10", "10 to 25", "25 to 50"], nulls="error"
    ).name("tier2")
    expr = size[labelled, size[1]]

    snapshot.assert_match(ImpalaCompiler.to_sql(expr), "out.sql")
