import numpy as np
import pytest
from pytest import param

pytest.importorskip("pyspark")


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t: t.age.count(),
            lambda t: len(t.age.dropna()),
            id='count',
        ),
        param(
            lambda t: t.age.sum(),
            lambda t: t.age.sum(),
            id='sum',
        ),
    ],
)
def test_aggregation_float_nulls(
    client,
    result_fn,
    expected_fn,
):
    table = client.table('null_table')
    df = table.compile().toPandas()

    expr = result_fn(table)
    result = expr.execute()

    expected = expected_fn(df)
    np.testing.assert_allclose(result, expected)
