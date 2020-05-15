import pytest

import ibis.expr.datatypes as dt
from ibis.tests.backends import Pandas, PySpark
from ibis.udf.vectorized import elementwise


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_elementwise_udf(backend, alltypes, df):
    @elementwise(input_type=[dt.double], output_type=dt.double)
    def add_one(s):
        return s + 1

    result = add_one(alltypes['double_col']).execute()
    expected = add_one.func(df['double_col'])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_elementwise_udf_with_output_type_in_list(backend, alltypes, df):
    """
    Test that a UDF's output type can be specified as a single datatype
    wrapped in a list. This is equivalent to a single datatype that is not in a
    list.
    """

    @elementwise(input_type=[dt.double], output_type=[dt.double])
    def add_one(s):
        return s + 1

    result = add_one(alltypes['double_col']).execute()
    expected = add_one.func(df['double_col'])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_valid_kwargs(backend, alltypes, df):
    # Test different forms of UDF definition

    @elementwise(input_type=[dt.double], output_type=dt.double)
    def foo1(v):
        # Basic UDF with kwargs
        return v + 1

    @elementwise(input_type=[dt.double], output_type=dt.double)
    def foo2(v, *, amount):
        # UDF with keyword only arguments
        return v + amount

    @elementwise(input_type=[dt.double], output_type=dt.double)
    def foo3(v, **kwargs):
        # UDF with kwargs
        return v + kwargs.get('amount', 1)

    result = alltypes.mutate(
        v1=foo1(alltypes['double_col']),
        v2=foo2(alltypes['double_col'], amount=1),
        v3=foo2(alltypes['double_col'], amount=2),
        v4=foo3(alltypes['double_col']),
        v5=foo3(alltypes['double_col'], amount=2),
        v6=foo3(alltypes['double_col'], amount=3),
    ).execute()

    expected = df.assign(
        v1=df['double_col'] + 1,
        v2=df['double_col'] + 1,
        v3=df['double_col'] + 2,
        v4=df['double_col'] + 1,
        v5=df['double_col'] + 2,
        v6=df['double_col'] + 3,
    )

    backend.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_invalid_kwargs(backend, alltypes):
    # Test different invalid UDFs

    with pytest.raises(TypeError, match=".*must be defined as keyword only.*"):

        @elementwise(input_type=[dt.double], output_type=dt.double)
        def foo1(v, amount):
            return v + 1
