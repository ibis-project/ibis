import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.tests.backends import Pandas, PySpark
from ibis.udf.vectorized import elementwise, reduction

pytestmark = pytest.mark.udf


@elementwise(input_type=[dt.double], output_type=dt.double)
def add_one(s):
    return s + 1


@reduction(input_type=[dt.double], output_type=dt.double)
def calc_mean(s):
    return s.mean()


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_elementwise_udf(backend, alltypes, df):
    result = add_one(alltypes['double_col']).execute()
    expected = add_one.func(df['double_col'])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_reduction_udf(backend, alltypes, df):
    result = calc_mean(alltypes['double_col']).execute()
    expected = df['double_col'].agg(calc_mean.func)
    assert result == expected


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_reduction_udf_aggregate(backend, alltypes, df):
    expr = alltypes.aggregate(tmp=calc_mean(alltypes.double_col))
    result = expr.execute()

    # Create a single-row single-column dataframe
    # (to match the output format of Ibis `aggregate`)
    expected = pd.DataFrame({'tmp': [df.double_col.mean()]})

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_reduction_udf_window(backend, alltypes, df):
    # Unbounded window
    window = ibis.window(
        group_by=[alltypes.string_col], order_by=[alltypes.id],
    )
    expr = alltypes.mutate(val=alltypes.double_col.mean().over(window))
    result = expr.execute().set_index('id').sort_index()

    gb = df.sort_values('id').groupby('string_col')
    column = gb.double_col.transform('mean')
    expected = df.assign(val=column).set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_output_type_in_list_invalid(backend, alltypes, df):
    # Test that an error is raised if UDF output type is wrapped in a list

    with pytest.raises(
        com.IbisTypeError,
        match="The output type of a UDF must be a single datatype.",
    ):

        @elementwise(input_type=[dt.double], output_type=[dt.double])
        def add_one(s):
            return s + 1


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_valid_kwargs(backend, alltypes, df):
    # Test different forms of UDF definition with keyword arguments

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
def test_valid_args(backend, alltypes, df):
    # Test different forms of UDF definition with *args

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo1(*args):
        return args[0] + len(args[1])

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo2(v, *args):
        return v + len(args[0])

    result = alltypes.mutate(
        v1=foo1(alltypes['double_col'], alltypes['string_col']),
        v2=foo2(alltypes['double_col'], alltypes['string_col']),
    ).execute()

    expected = df.assign(
        v1=df['double_col'] + len(df['string_col']),
        v2=df['double_col'] + len(df['string_col']),
    )

    backend.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_valid_args_and_kwargs(backend, alltypes, df):
    # Test UDFs with both *args and keyword arguments

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo1(*args, amount):
        # UDF with *args and a keyword-only argument
        return args[0] + len(args[1]) + amount

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo2(*args, **kwargs):
        # UDF with *args and **kwargs
        return args[0] + len(args[1]) + kwargs.get('amount', 1)

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo3(v, *args, amount):
        # UDF with an explicit positional argument, *args, and a keyword-only
        # argument
        return v + len(args[0]) + amount

    @elementwise(input_type=[dt.double, dt.string], output_type=dt.double)
    def foo4(v, *args, **kwargs):
        # UDF with an explicit positional argument, *args, and **kwargs
        return v + len(args[0]) + kwargs.get('amount', 1)

    result = alltypes.mutate(
        v1=foo1(alltypes['double_col'], alltypes['string_col'], amount=2),
        v2=foo2(alltypes['double_col'], alltypes['string_col'], amount=2),
        v3=foo3(alltypes['double_col'], alltypes['string_col'], amount=2),
        v4=foo4(alltypes['double_col'], alltypes['string_col'], amount=2),
    ).execute()

    expected = df.assign(
        v1=df['double_col'] + len(df['string_col']) + 2,
        v2=df['double_col'] + len(df['string_col']) + 2,
        v3=df['double_col'] + len(df['string_col']) + 2,
        v4=df['double_col'] + len(df['string_col']) + 2,
    )

    backend.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_invalid_kwargs(backend, alltypes):
    # Test that defining a UDF with a non-column argument that is not a
    # keyword argument raises an error

    with pytest.raises(TypeError, match=".*must be defined as keyword only.*"):

        @elementwise(input_type=[dt.double], output_type=dt.double)
        def foo1(v, amount):
            return v + 1
