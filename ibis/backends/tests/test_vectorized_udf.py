from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.legacy.udf.vectorized import analytic, elementwise, reduction

pytestmark = pytest.mark.notimpl(["druid", "oracle"])


def _format_udf_return_type(func, result_formatter):
    """Call the given udf and return its result according to the given format
    (e.g. in the form of a list, pd.Series, np.array, etc.)"""

    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result_formatter(result)

    return _wrapper


def _format_struct_udf_return_type(func, result_formatter):
    """Call the given struct udf and return its result according to the given
    format (e.g. in the form of a list, pd.Series, np.array, etc.)"""

    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result_formatter(*result)

    return _wrapper


# elementwise UDF
def add_one(s):
    assert isinstance(s, pd.Series)
    return s + 1


def create_add_one_udf(result_formatter, id):
    @elementwise(input_type=[dt.double], output_type=dt.double)
    def add_one_legacy(s):
        return result_formatter(add_one(s))

    @ibis.udf.scalar.pandas
    def add_one_udf(s: float) -> float:
        return result_formatter(add_one(s))

    yield param(add_one_legacy, id=f"add_one_legacy_{id}")
    yield param(
        add_one_udf,
        marks=[pytest.mark.notimpl(["pandas", "dask"])],
        id=f"add_one_modern_{id}",
    )


add_one_udfs = [
    *create_add_one_udf(result_formatter=lambda v: v, id="series"),
    *create_add_one_udf(result_formatter=lambda v: np.array(v), id="array"),
    *create_add_one_udf(result_formatter=lambda v: list(v), id="list"),
]


# analytic UDF
def calc_zscore(s):
    assert isinstance(s, pd.Series)
    return (s - s.mean()) / s.std()


def create_calc_zscore_udf(result_formatter):
    return analytic(input_type=[dt.double], output_type=dt.double)(
        _format_udf_return_type(calc_zscore, result_formatter)
    )


calc_zscore_udfs = [
    create_calc_zscore_udf(result_formatter=lambda v: v),  # pd.Series,
    create_calc_zscore_udf(result_formatter=lambda v: np.array(v)),  # np.array,
    create_calc_zscore_udf(result_formatter=lambda v: list(v)),  # list,
]


@reduction(input_type=[dt.double], output_type=dt.double)
def calc_mean(s):
    assert isinstance(s, (np.ndarray, pd.Series))
    return s.mean()


# elementwise multi-column UDF
def add_one_struct(v):
    assert isinstance(v, pd.Series)
    return v + 1, v + 2


def create_add_one_struct_udf(result_formatter):
    return elementwise(
        input_type=[dt.double],
        output_type=dt.Struct({"col1": dt.double, "col2": dt.double}),
    )(_format_struct_udf_return_type(add_one_struct, result_formatter))


add_one_struct_udfs = [
    param(
        create_add_one_struct_udf(result_formatter=lambda v1, v2: (v1, v2)),
        id="tuple_of_series",
    ),
    param(
        create_add_one_struct_udf(result_formatter=lambda v1, v2: [v1, v2]),
        id="list_of_series",
    ),
    param(
        create_add_one_struct_udf(
            result_formatter=lambda v1, v2: (np.array(v1), np.array(v2))
        ),
        id="tuple_of_ndarray",
    ),
    param(
        create_add_one_struct_udf(
            result_formatter=lambda v1, v2: [np.array(v1), np.array(v2)]
        ),
        id="list_of_ndarray",
    ),
    param(
        create_add_one_struct_udf(
            result_formatter=lambda v1, v2: np.array([np.array(v1), np.array(v2)])
        ),
        id="ndarray_of_ndarray",
    ),
    param(
        create_add_one_struct_udf(
            result_formatter=lambda v1, v2: pd.DataFrame({"col1": v1, "col2": v2})
        ),
        id="dataframe",
    ),
]


@elementwise(
    input_type=[dt.double],
    output_type=dt.Struct({"double_col": dt.double, "col2": dt.double}),
)
def overwrite_struct_elementwise(v):
    assert isinstance(v, pd.Series)
    return v + 1, v + 2


@elementwise(
    input_type=[dt.double],
    output_type=dt.Struct(
        {"double_col": dt.double, "col2": dt.double, "float_col": dt.double}
    ),
)
def multiple_overwrite_struct_elementwise(v):
    assert isinstance(v, pd.Series)
    return v + 1, v + 2, v + 3


@analytic(
    input_type=[dt.double, dt.double],
    output_type=dt.Struct({"double_col": dt.double, "demean_weight": dt.double}),
)
def overwrite_struct_analytic(v, w):
    assert isinstance(v, pd.Series)
    assert isinstance(w, pd.Series)
    return v - v.mean(), w - w.mean()


# analytic multi-column UDF
def demean_struct(v, w):
    assert isinstance(v, pd.Series)
    assert isinstance(w, pd.Series)
    return v - v.mean(), w - w.mean()


def create_demean_struct_udf(result_formatter):
    return analytic(
        input_type=[dt.double, dt.double],
        output_type=dt.Struct({"demean": dt.double, "demean_weight": dt.double}),
    )(_format_struct_udf_return_type(demean_struct, result_formatter))


demean_struct_udfs = [
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    ),  # tuple of pd.Series,
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: [v1, v2]
    ),  # list of pd.Series,
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: (np.array(v1), np.array(v2))
    ),  # tuple of np.array,
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: [np.array(v1), np.array(v2)]
    ),  # list of np.array,
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: np.array([np.array(v1), np.array(v2)])
    ),  # np.array of np.array,
    create_demean_struct_udf(
        result_formatter=lambda v1, v2: pd.DataFrame(
            {"demean": v1, "demean_weight": v2}
        )
    ),  # pd.DataFrame,
]


# reduction multi-column UDF
def mean_struct(v, w):
    assert isinstance(v, (np.ndarray, pd.Series))
    assert isinstance(w, (np.ndarray, pd.Series))
    return v.mean(), w.mean()


def create_mean_struct_udf(result_formatter):
    return reduction(
        input_type=[dt.double, dt.int64],
        output_type=dt.Struct({"mean": dt.double, "mean_weight": dt.double}),
    )(_format_struct_udf_return_type(mean_struct, result_formatter))


mean_struct_udfs = [
    create_mean_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    ),  # tuple of scalar,
    create_mean_struct_udf(result_formatter=lambda v1, v2: [v1, v2]),  # list of scalar,
    create_mean_struct_udf(
        result_formatter=lambda v1, v2: np.array([v1, v2])
    ),  # np.array of scalar
]


@reduction(
    input_type=[dt.double, dt.int64],
    output_type=dt.Struct({"double_col": dt.double, "mean_weight": dt.double}),
)
def overwrite_struct_reduction(v, w):
    assert isinstance(v, (np.ndarray, pd.Series))
    assert isinstance(w, (np.ndarray, pd.Series))
    return v.mean(), w.mean()


@reduction(
    input_type=[dt.double],
    output_type=dt.Array(dt.double),
)
def quantiles(series, *, quantiles):
    return series.quantile(quantiles)


@pytest.mark.parametrize(
    "udf", create_add_one_udf(result_formatter=lambda v: v, id="series")
)
def test_elementwise_udf(udf_backend, udf_alltypes, udf_df, udf):
    expr = udf(udf_alltypes["double_col"])
    result = expr.execute()

    expected_func = getattr(expr.op(), "__func__", getattr(udf, "func", None))
    assert (
        expected_func is not None
    ), f"neither __func__ nor func attributes found on {udf} or expr object"

    expected = expected_func(udf_df["double_col"])
    udf_backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("udf", add_one_udfs)
def test_elementwise_udf_mutate(udf_backend, udf_alltypes, udf_df, udf):
    udf_expr = udf(udf_alltypes["double_col"])
    expr = udf_alltypes.mutate(incremented=udf_expr)
    result = expr.execute()

    expected_func = getattr(udf_expr.op(), "__func__", getattr(udf, "func", None))
    assert (
        expected_func is not None
    ), f"neither __func__ nor func attributes found on {udf} or expr object"

    expected = udf_df.assign(incremented=expected_func(udf_df["double_col"]))
    udf_backend.assert_series_equal(result["incremented"], expected["incremented"])


@pytest.mark.notimpl(["pyspark"])
def test_analytic_udf(udf_backend, udf_alltypes, udf_df):
    calc_zscore_udf = create_calc_zscore_udf(result_formatter=lambda v: v)
    result = calc_zscore_udf(udf_alltypes["double_col"]).execute()
    expected = calc_zscore_udf.func(udf_df["double_col"])
    udf_backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("udf", calc_zscore_udfs)
@pytest.mark.notimpl(["pyspark"])
def test_analytic_udf_mutate(udf_backend, udf_alltypes, udf_df, udf):
    expr = udf_alltypes.mutate(zscore=udf(udf_alltypes["double_col"]))
    result = expr.execute()

    expected = udf_df.assign(zscore=udf.func(udf_df["double_col"]))

    udf_backend.assert_series_equal(result["zscore"], expected["zscore"])


def test_reduction_udf(udf_alltypes, udf_df):
    result = calc_mean(udf_alltypes["double_col"]).execute()
    expected = udf_df["double_col"].mean()
    assert result == expected


def test_reduction_udf_array_return_type(udf_backend, udf_alltypes, udf_df):
    """Tests reduction UDF returning an array."""
    qs = [0.25, 0.75]
    expr = udf_alltypes.mutate(q=quantiles(udf_alltypes["int_col"], quantiles=qs))
    result = expr.execute()

    expected = udf_df.assign(
        q=pd.Series([quantiles.func(udf_df["int_col"], quantiles=qs)])
        .repeat(len(udf_df))
        .reset_index(drop=True)
    )
    udf_backend.assert_frame_equal(result, expected)


def test_reduction_udf_on_empty_data(udf_backend, udf_alltypes):
    """Test that summarization can handle empty data."""
    # First filter down to zero rows
    t = udf_alltypes[udf_alltypes["int_col"] > np.inf]
    result = t.group_by("year").aggregate(mean=calc_mean(t["int_col"])).execute()
    expected = pd.DataFrame({"year": [], "mean": []})
    # We check that the result is an empty DataFrame,
    # rather than an error.
    udf_backend.assert_frame_equal(result, expected, check_dtype=False)


def test_output_type_in_list_invalid():
    # Test that an error is raised if UDF output type is wrapped in a list

    with pytest.raises(
        com.IbisTypeError,
        match="The output type of a UDF must be a single datatype.",
    ):

        @elementwise(input_type=[dt.double], output_type=[dt.double])
        def _(s):
            return s + 1


def test_valid_kwargs(udf_backend, udf_alltypes, udf_df):
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
        return v + kwargs.get("amount", 1)

    result = udf_alltypes.mutate(
        v1=foo1(udf_alltypes["double_col"]),
        v2=foo2(udf_alltypes["double_col"], amount=1),
        v3=foo2(udf_alltypes["double_col"], amount=2),
        v4=foo3(udf_alltypes["double_col"]),
        v5=foo3(udf_alltypes["double_col"], amount=2),
        v6=foo3(udf_alltypes["double_col"], amount=3),
    ).execute()

    expected = udf_df.assign(
        v1=udf_df["double_col"] + 1,
        v2=udf_df["double_col"] + 1,
        v3=udf_df["double_col"] + 2,
        v4=udf_df["double_col"] + 1,
        v5=udf_df["double_col"] + 2,
        v6=udf_df["double_col"] + 3,
    )

    udf_backend.assert_frame_equal(result, expected)


def test_valid_args(udf_backend, udf_alltypes, udf_df):
    # Test different forms of UDF definition with *args

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo1(*args):
        return args[0] + args[1]

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo2(v, *args):
        return v + args[0]

    result = udf_alltypes.mutate(
        v1=foo1(udf_alltypes["double_col"], udf_alltypes["int_col"]),
        v2=foo2(udf_alltypes["double_col"], udf_alltypes["int_col"]),
    ).execute()

    expected = udf_df.assign(
        v1=udf_df["double_col"] + udf_df["int_col"],
        v2=udf_df["double_col"] + udf_df["int_col"],
    )

    udf_backend.assert_frame_equal(result, expected)


def test_valid_args_and_kwargs(udf_backend, udf_alltypes, udf_df):
    # Test UDFs with both *args and keyword arguments

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo1(*args, amount):
        # UDF with *args and a keyword-only argument
        return args[0] + args[1] + amount

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo2(*args, **kwargs):
        # UDF with *args and **kwargs
        return args[0] + args[1] + kwargs.get("amount", 1)

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo3(v, *args, amount):
        # UDF with an explicit positional argument, *args, and a keyword-only
        # argument
        return v + args[0] + amount

    @elementwise(input_type=[dt.double, dt.int32], output_type=dt.double)
    def foo4(v, *args, **kwargs):
        # UDF with an explicit positional argument, *args, and **kwargs
        return v + args[0] + kwargs.get("amount", 1)

    result = udf_alltypes.mutate(
        v1=foo1(udf_alltypes["double_col"], udf_alltypes["int_col"], amount=2),
        v2=foo2(udf_alltypes["double_col"], udf_alltypes["int_col"], amount=2),
        v3=foo3(udf_alltypes["double_col"], udf_alltypes["int_col"], amount=2),
        v4=foo4(udf_alltypes["double_col"], udf_alltypes["int_col"], amount=2),
    ).execute()

    expected = udf_df.assign(
        v1=udf_df["double_col"] + udf_df["int_col"] + 2,
        v2=udf_df["double_col"] + udf_df["int_col"] + 2,
        v3=udf_df["double_col"] + udf_df["int_col"] + 2,
        v4=udf_df["double_col"] + udf_df["int_col"] + 2,
    )

    udf_backend.assert_frame_equal(result, expected)


def test_invalid_kwargs():
    # Test that defining a UDF with a non-column argument that is not a
    # keyword argument raises an error

    with pytest.raises(TypeError, match=".*must be defined as keyword only.*"):

        @elementwise(input_type=[dt.double], output_type=dt.double)
        def _(v, _):
            return v + 1


@pytest.mark.parametrize("udf", add_one_struct_udfs)
def test_elementwise_udf_destruct(udf_backend, udf_alltypes, udf):
    result = udf_alltypes.mutate(
        udf(udf_alltypes["double_col"]).destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        col1=udf_alltypes["double_col"] + 1,
        col2=udf_alltypes["double_col"] + 2,
    ).execute()

    udf_backend.assert_frame_equal(result, expected)


def test_elementwise_udf_overwrite_destruct(udf_backend, udf_alltypes):
    result = udf_alltypes.mutate(
        overwrite_struct_elementwise(udf_alltypes["double_col"]).destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        double_col=udf_alltypes["double_col"] + 1,
        col2=udf_alltypes["double_col"] + 2,
    ).execute()

    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    udf_backend.assert_frame_equal(result, expected, check_like=True)


def test_elementwise_udf_overwrite_destruct_and_assign(udf_backend, udf_alltypes):
    result = (
        udf_alltypes.mutate(
            overwrite_struct_elementwise(udf_alltypes["double_col"]).destructure()
        )
        .mutate(col3=udf_alltypes["int_col"] * 3)
        .execute()
    )

    expected = udf_alltypes.mutate(
        double_col=udf_alltypes["double_col"] + 1,
        col2=udf_alltypes["double_col"] + 2,
        col3=udf_alltypes["int_col"] * 3,
    ).execute()

    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    udf_backend.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.xfail_version(pyspark=["pyspark<3.1"])
@pytest.mark.parametrize("method", ["destructure", "unpack"])
@pytest.mark.skip("dask")
def test_elementwise_udf_destructure_exact_once(udf_alltypes, method, tmp_path):
    @elementwise(
        input_type=[dt.double],
        output_type=dt.Struct({"col1": dt.double, "col2": dt.double}),
    )
    def add_one_struct_exact_once(v):
        key = v.iloc[0]
        path = tmp_path / str(key)
        assert not path.exists()
        path.touch()
        return v + 1, v + 2

    struct = add_one_struct_exact_once(udf_alltypes["id"])

    if method == "destructure":
        expr = udf_alltypes.mutate(struct.destructure())
    elif method == "unpack":
        expr = udf_alltypes.mutate(struct=struct).unpack("struct")
    else:
        raise ValueError(f"Invalid method {method}")
    result = expr.execute()

    assert len(result) > 0


def test_elementwise_udf_multiple_overwrite_destruct(udf_backend, udf_alltypes):
    result = udf_alltypes.mutate(
        multiple_overwrite_struct_elementwise(udf_alltypes["double_col"]).destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        double_col=udf_alltypes["double_col"] + 1,
        col2=udf_alltypes["double_col"] + 2,
        float_col=udf_alltypes["double_col"] + 3,
    ).execute()

    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    udf_backend.assert_frame_equal(result, expected, check_like=True)


def test_elementwise_udf_named_destruct(udf_alltypes):
    """Test error when assigning name to a destruct column."""

    add_one_struct_udf = create_add_one_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    )
    with pytest.raises(com.IbisTypeError, match=r"Unable to infer"):
        udf_alltypes.mutate(
            new_struct=add_one_struct_udf(udf_alltypes["double_col"]).destructure()
        )


def test_elementwise_udf_struct(udf_backend, udf_alltypes):
    add_one_struct_udf = create_add_one_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    )
    result = udf_alltypes.mutate(
        new_col=add_one_struct_udf(udf_alltypes["double_col"])
    ).execute()
    result = result.assign(
        col1=result["new_col"].apply(lambda x: x["col1"]),
        col2=result["new_col"].apply(lambda x: x["col2"]),
    )
    result = result.drop("new_col", axis=1)
    expected = udf_alltypes.mutate(
        col1=udf_alltypes["double_col"] + 1,
        col2=udf_alltypes["double_col"] + 2,
    ).execute()

    udf_backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize("udf", demean_struct_udfs)
@pytest.mark.notimpl(["pyspark"])
@pytest.mark.broken(["dask"], strict=False)
def test_analytic_udf_destruct(udf_backend, udf_alltypes, udf):
    w = ibis.window(preceding=None, following=None, group_by="year")

    result = udf_alltypes.mutate(
        udf(udf_alltypes["double_col"], udf_alltypes["int_col"]).over(w).destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        demean=udf_alltypes["double_col"] - udf_alltypes["double_col"].mean().over(w),
        demean_weight=udf_alltypes["int_col"] - udf_alltypes["int_col"].mean().over(w),
    ).execute()
    udf_backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["pyspark"])
def test_analytic_udf_destruct_no_group_by(udf_backend, udf_alltypes):
    w = ibis.window(preceding=None, following=None)

    demean_struct_udf = create_demean_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    )
    result = udf_alltypes.mutate(
        demean_struct_udf(udf_alltypes["double_col"], udf_alltypes["int_col"])
        .over(w)
        .destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        demean=udf_alltypes["double_col"] - udf_alltypes["double_col"].mean().over(w),
        demean_weight=udf_alltypes["int_col"] - udf_alltypes["int_col"].mean().over(w),
    ).execute()

    udf_backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["pyspark"])
@pytest.mark.xfail_version(dask=["pandas>=2"])
def test_analytic_udf_destruct_overwrite(udf_backend, udf_alltypes):
    w = ibis.window(preceding=None, following=None, group_by="year")

    result = udf_alltypes.mutate(
        overwrite_struct_analytic(udf_alltypes["double_col"], udf_alltypes["int_col"])
        .over(w)
        .destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        double_col=udf_alltypes["double_col"]
        - udf_alltypes["double_col"].mean().over(w),
        demean_weight=udf_alltypes["int_col"] - udf_alltypes["int_col"].mean().over(w),
    ).execute()

    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    udf_backend.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize("udf", mean_struct_udfs)
@pytest.mark.notimpl(["pyspark"])
def test_reduction_udf_destruct_group_by(udf_backend, udf_alltypes, udf):
    result = (
        udf_alltypes.group_by("year")
        .aggregate(
            udf(udf_alltypes["double_col"], udf_alltypes["int_col"]).destructure()
        )
        .execute()
    ).sort_values("year")

    expected = (
        udf_alltypes.group_by("year")
        .aggregate(
            mean=udf_alltypes["double_col"].mean(),
            mean_weight=udf_alltypes["int_col"].mean(),
        )
        .execute()
    ).sort_values("year")

    udf_backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["pyspark"])
def test_reduction_udf_destruct_no_group_by(udf_backend, udf_alltypes):
    mean_struct_udf = create_mean_struct_udf(result_formatter=lambda v1, v2: (v1, v2))
    result = udf_alltypes.aggregate(
        mean_struct_udf(
            udf_alltypes["double_col"], udf_alltypes["int_col"]
        ).destructure()
    ).execute()

    expected = udf_alltypes.aggregate(
        mean=udf_alltypes["double_col"].mean(),
        mean_weight=udf_alltypes["int_col"].mean(),
    ).execute()
    udf_backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["pyspark"])
def test_reduction_udf_destruct_no_group_by_overwrite(udf_backend, udf_alltypes):
    result = udf_alltypes.aggregate(
        overwrite_struct_reduction(
            udf_alltypes["double_col"], udf_alltypes["int_col"]
        ).destructure()
    ).execute()

    expected = udf_alltypes.aggregate(
        double_col=udf_alltypes["double_col"].mean(),
        mean_weight=udf_alltypes["int_col"].mean(),
    ).execute()
    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    udf_backend.assert_frame_equal(result, expected, check_like=True)


# TODO - windowing - #2553
@pytest.mark.notimpl(["dask", "pyspark"])
def test_reduction_udf_destruct_window(udf_backend, udf_alltypes):
    win = ibis.window(
        preceding=ibis.interval(hours=2),
        following=0,
        group_by="year",
        order_by="timestamp_col",
    )
    mean_struct_udf = create_mean_struct_udf(result_formatter=lambda v1, v2: (v1, v2))

    result = udf_alltypes.mutate(
        mean_struct_udf(udf_alltypes["double_col"], udf_alltypes["int_col"])
        .over(win)
        .destructure()
    ).execute()

    expected = udf_alltypes.mutate(
        mean=udf_alltypes["double_col"].mean().over(win),
        mean_weight=udf_alltypes["int_col"].mean().over(win),
    ).execute()

    udf_backend.assert_frame_equal(result, expected)
