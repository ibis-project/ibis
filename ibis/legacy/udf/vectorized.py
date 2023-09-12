"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

import ibis.expr.datatypes as dt
import ibis.legacy.udf.validate as v
from ibis.expr.operations import (
    AnalyticVectorizedUDF,
    ElementWiseVectorizedUDF,
    ReductionVectorizedUDF,
)

if TYPE_CHECKING:
    import pandas as pd


def _coerce_to_dict(
    data: list | np.ndarray | pd.Series,
    output_type: dt.Struct,
    index: pd.Index | None = None,
) -> tuple:
    """Coerce the following shapes to a tuple.

    - [](`list`)
    - `np.ndarray`
    - `pd.Series`
    """
    return dict(zip(output_type.names, data))


def _coerce_to_np_array(
    data: list | np.ndarray | pd.Series,
    output_type: dt.Struct,
    index: pd.Index | None = None,
) -> np.ndarray:
    """Coerce the following shapes to an np.ndarray.

    - [](`list`)
    - `np.ndarray`
    - `pd.Series`
    """
    return np.array(data)


def _coerce_to_series(
    data: list | np.ndarray | pd.Series,
    output_type: dt.DataType,
    original_index: pd.Index | None = None,
) -> pd.Series:
    """Coerce the following shapes to a Series.

    This method does NOT always return a new Series. If a Series is
    passed in, this method will return the original object.

    - [](`list`)
    - `np.ndarray`
    - `pd.Series`

    Note:

    Parameters
    ----------
    data
        Input
    output_type
        The type of the output
    original_index
        Optional parameter containing the index of the output

    Returns
    -------
    pd.Series
        Output Series
    """
    import pandas as pd

    if isinstance(data, (list, np.ndarray)):
        result = pd.Series(data)
    elif isinstance(data, pd.Series):
        result = data
    else:
        # This case is a non-vector elementwise or analytic UDF that should
        # not be coerced to a Series.
        return data
    if original_index is not None:
        result.index = original_index
    return result


def _coerce_to_dataframe(
    data: Any,
    output_type: dt.Struct,
    original_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Coerce the following shapes to a DataFrame.

    This method does NOT always return a new DataFrame. If a DataFrame is
    passed in, this method will return the original object.

    The following shapes are allowed:

    - A list/tuple of Series
    - A list/tuple np.ndarray
    - A list/tuple of scalars
    - A Series of list/tuple
    - pd.DataFrame

    Note:

    Parameters
    ----------
    data
        Input
    output_type
        A Struct containing the names and types of the output
    original_index
        Optional parameter containing the index of the output

    Returns
    -------
    pd.DataFrame
        Output DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> _coerce_to_dataframe(
    ...     pd.DataFrame({"a": [1, 2, 3]}), dt.Struct(dict(b="int32"))
    ... )  # noqa: E501
       b
    0  1
    1  2
    2  3
    >>> _coerce_to_dataframe(
    ...     pd.Series([[1, 2, 3]]), dt.Struct(dict.fromkeys("abc", "int32"))
    ... )  # noqa: E501
       a  b  c
    0  1  2  3
    >>> _coerce_to_dataframe(
    ...     pd.Series([range(3), range(3)]), dt.Struct(dict.fromkeys("abc", "int32"))
    ... )  # noqa: E501
       a  b  c
    0  0  1  2
    1  0  1  2
    >>> _coerce_to_dataframe(
    ...     [pd.Series(x) for x in [1, 2, 3]], dt.Struct(dict.fromkeys("abc", "int32"))
    ... )  # noqa: E501
       a  b  c
    0  1  2  3
    >>> _coerce_to_dataframe(
    ...     [1, 2, 3], dt.Struct(dict.fromkeys("abc", "int32"))
    ... )  # noqa: E501
       a  b  c
    0  1  2  3
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        result = data
    elif isinstance(data, pd.Series):
        if not len(data):
            result = data.to_frame()
        else:
            num_cols = len(data.iloc[0])
            series = [data.apply(lambda t, i=i: t[i]) for i in range(num_cols)]
            result = pd.concat(series, axis=1)
    elif isinstance(data, (tuple, list, np.ndarray)):
        if isinstance(data[0], pd.Series):
            result = pd.concat(data, axis=1)
        elif isinstance(data[0], np.ndarray):
            result = pd.concat([pd.Series(v) for v in data], axis=1)
        else:
            # Promote scalar to Series
            result = pd.concat([pd.Series([v]) for v in data], axis=1)
    else:
        raise ValueError(f"Cannot coerce to DataFrame: {data}")

    result.columns = output_type.names
    if original_index is not None:
        result.index = original_index
    return result


class UserDefinedFunction:
    """Class representing a user defined function.

    This class Implements __call__ that returns an ibis expr for the
    UDF.
    """

    def __init__(self, func, func_type, input_type, output_type):
        v.validate_input_type(input_type, func)
        v.validate_output_type(output_type)

        self.func = func
        self.func_type = func_type
        self.input_type = list(map(dt.dtype, input_type))
        self.output_type = dt.dtype(output_type)
        self.coercion_fn = self._get_coercion_function()

    def _get_coercion_function(self):
        """Return the appropriate function to coerce the result of the UDF."""
        if self.output_type.is_struct():
            # Case 1: Struct output, non-reduction UDF -> coerce to DataFrame
            if (
                self.func_type is ElementWiseVectorizedUDF
                or self.func_type is AnalyticVectorizedUDF
            ):
                return _coerce_to_dataframe
            else:
                # Case 2: Struct output, reduction UDF -> coerce to dictionary
                return _coerce_to_dict
        # Case 3: Vector output, non-reduction UDF -> coerce to Series
        elif (
            self.func_type is ElementWiseVectorizedUDF
            or self.func_type is AnalyticVectorizedUDF
        ):
            return _coerce_to_series
        # Case 4: Array output type, reduction UDF -> coerce to np.ndarray
        elif self.output_type.is_array():
            return _coerce_to_np_array
        else:
            # Case 5: Default, do nothing (e.g. reduction UDF returning
            # len-0 value such as a single integer or float).
            return None

    def __call__(self, *args, **kwargs):
        # kwargs cannot be part of the node object because it can contain
        # unhashable object, e.g., list.
        # Here, we keep the node hashable by creating a closure that contains
        # kwargs.
        @functools.wraps(self.func)
        def func(*args):
            # If cols are pd.Series, then we save and restore the index.
            saved_index = getattr(args[0], "index", None)
            result = self.func(*args, **kwargs)
            if self.coercion_fn:
                # coercion function signature must take result, output type,
                # and optionally the index
                result = self.coercion_fn(result, self.output_type, saved_index)
            return result

        op = self.func_type(
            func=func,
            func_args=args,
            input_type=self.input_type,
            return_type=self.output_type,
        )

        return op.to_expr()


def _udf_decorator(node_type, input_type, output_type):
    def wrapper(func):
        return UserDefinedFunction(func, node_type, input_type, output_type)

    return wrapper


def analytic(input_type, output_type):
    """Define an analytic UDF that produces the same of rows as the input.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.legacy.udf.vectorized import analytic
    >>> @analytic(input_type=[dt.double], output_type=dt.double)
    ... def zscore(series):  # note the use of aggregate functions
    ...     return (series - series.mean()) / series.std()
    ...

    Define and use an UDF with multiple return columns:

    >>> @analytic(
    ...     input_type=[dt.double],
    ...     output_type=dt.Struct(dict(demean="double", zscore="double")),
    ... )
    ... def demean_and_zscore(v):
    ...     mean = v.mean()
    ...     std = v.std()
    ...     return v - mean, (v - mean) / std
    >>>
    >>> win = ibis.window(preceding=None, following=None, group_by="key")
    >>> # add two columns "demean" and "zscore"
    >>> table = table.mutate(  # quartodoc: +SKIP # doctest: +SKIP
    ...     demean_and_zscore(table["v"]).over(win).destructure()
    ... )
    """
    return _udf_decorator(AnalyticVectorizedUDF, input_type, output_type)


def elementwise(input_type, output_type):
    """Define a UDF that operates element-wise on a Pandas Series.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.legacy.udf.vectorized import elementwise
    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series):
    ...     return series.str.len() * 2
    ...

    Define an UDF with non-column parameters:

    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series, *, times):
    ...     return series.str.len() * times
    ...

    Define and use an UDF with multiple return columns:

    >>> @elementwise(
    ...     input_type=[dt.string],
    ...     output_type=dt.Struct(dict(year=dt.string, monthday=dt.string)),
    ... )
    ... def year_monthday(date):
    ...     return date.str.slice(0, 4), date.str.slice(4, 8)
    >>>
    >>> # add two columns "year" and "monthday"
    >>> table = table.mutate(
    ...     year_monthday(table["date"]).destructure()
    ... )  # quartodoc: +SKIP # doctest: +SKIP
    """
    return _udf_decorator(ElementWiseVectorizedUDF, input_type, output_type)


def reduction(input_type, output_type):
    """Define a UDF reduction function that produces 1 row of output for N rows of input.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.legacy.udf.vectorized import reduction
    >>> @reduction(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length_agg(series, **kwargs):
    ...     return (series.str.len() * 2).sum()
    ...

    Define and use an UDF with multiple return columns:

    >>> @reduction(
    ...     input_type=[dt.double],
    ...     output_type=dt.Struct(dict(mean="double", std="double")),
    ... )
    ... def mean_and_std(v):
    ...     return v.mean(), v.std()
    >>>
    >>> # create aggregation columns "mean" and "std"
    >>> table = table.group_by("key").aggregate(  # quartodoc: +SKIP # doctest: +SKIP
    ...     mean_and_std(table["v"]).destructure()
    ... )
    """
    return _udf_decorator(ReductionVectorizedUDF, input_type, output_type)
