import pandas as pd
import pytest
from pytest import param


@pytest.mark.parametrize(
    "how",
    [
        param(
            "inner",
            marks=pytest.mark.notimpl(["datafusion", "impala"]),
        ),
        param(
            "left",
            marks=pytest.mark.notimpl(["datafusion", "impala"]),
        ),
        param(
            "right",
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "datafusion",
                    "duckdb",
                    "mysql",
                    "postgres",
                    "sqlite",
                    "impala",
                ]
            ),
        ),
        param(
            "outer",
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "datafusion",
                    "duckdb",
                    "mysql",
                    "postgres",
                    "sqlite",
                    "impala",
                ]
            ),
        ),
        param(
            "semi",
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "dask",
                    "datafusion",
                    "duckdb",
                    "impala",
                    "mysql",
                    "pandas",
                    "postgres",
                    "pyspark",
                    "sqlite",
                ]
            ),
        ),
        param(
            "anti",
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "dask",
                    "datafusion",
                    "duckdb",
                    "impala",
                    "mysql",
                    "pandas",
                    "postgres",
                    "pyspark",
                    "sqlite",
                ]
            ),
        ),
    ],
)
def test_join_project_left_table(backend, con, batting, awards_players, how):

    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == 'NL'].drop(
        ['yearID', 'lgID']
    )

    left_df = left.execute()
    right_df = right.execute()
    predicate = ['playerID']
    result_order = ['playerID', 'yearID', 'lgID', 'stint']
    expr = left.join(right, predicate, how=how)[left]
    result = expr.execute().sort_values(result_order)

    joined = pd.merge(
        left_df, right_df, how=how, on=predicate, suffixes=('', '_y')
    ).sort_values(result_order)
    expected = joined[list(left.columns)]

    backend.assert_frame_equal(
        result[expected.columns], expected, check_like=True
    )
