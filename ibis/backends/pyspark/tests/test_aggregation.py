import pytest
from pytest import param

import ibis

pytest.importorskip("pyspark")


@pytest.fixture
def treat_nan_as_null():
    treat_nan_as_null = ibis.options.pyspark.treat_nan_as_null
    ibis.options.pyspark.treat_nan_as_null = True
    try:
        yield
    finally:
        ibis.options.pyspark.treat_nan_as_null = treat_nan_as_null


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
    treat_nan_as_null,
):
    table = client.table('null_table')
    df = table.compile().toPandas()

    expr = result_fn(table)
    result = expr.execute()

    expected = expected_fn(df)
    assert pytest.approx(expected) == result
