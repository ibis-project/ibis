import pandas as pd
import pytest
from pytest import param

import ibis


@pytest.mark.parametrize(
    "distinct",
    [param(False, id="all"), param(True, id="distinct")],
)
@pytest.mark.notimpl(["datafusion"])
def test_union(backend, alltypes, df, distinct):
    default_limit = ibis.options.sql.default_limit
    ibis.options.sql.default_limit = 15000
    try:
        expr = alltypes.union(alltypes, distinct=distinct).sort_by("id")
        result = expr.execute().reset_index(drop=True)
        expected = df if distinct else pd.concat([df, df], axis=0)

        backend.assert_frame_equal(
            result,
            expected.sort_values("id").reset_index(drop=True),
        )
    finally:
        ibis.options.sql.default_limit = default_limit


@pytest.mark.parametrize(
    "distinct",
    [
        param(False, id="all"),
        param(
            True,
            marks=pytest.mark.notimpl(
                ["duckdb", "postgres"],
                reason="Result order not guaranteed when distinct=True",
            ),
            id="distinct",
        ),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_union_no_sort(backend, alltypes, df, distinct):
    result = alltypes.union(alltypes, distinct=distinct).execute()
    expected = df if distinct else pd.concat([df, df], axis=0)
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["clickhouse", "dask", "pandas", "sqlite"],
                reason="backend doesn't support INTERSECT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["impala"])
def test_intersect(backend, alltypes, df, distinct):
    expr = alltypes.intersect(alltypes, distinct=distinct).sort_by("id")
    result = expr.execute()
    expected = df
    backend.assert_frame_equal(
        result,
        expected.sort_values("id").reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["clickhouse", "dask", "pandas", "sqlite"],
                reason="backend doesn't support EXCEPT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["impala"])
def test_difference(backend, alltypes, distinct):
    expr = alltypes.difference(alltypes, distinct=distinct)
    result = expr.execute()

    dtypes = {
        column: dtype.to_pandas() for column, dtype in expr.schema().items()
    }
    expected = pd.DataFrame(columns=alltypes.columns).astype(dtypes)
    backend.assert_frame_equal(result, expected)
