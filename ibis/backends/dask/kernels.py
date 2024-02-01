from __future__ import annotations

import dask.dataframe as dd
import numpy as np

import ibis.backends.pandas.kernels as pandas_kernels
import ibis.expr.operations as ops

generic = pandas_kernels.generic.copy()
columnwise = pandas_kernels.columnwise.copy()
elementwise = pandas_kernels.elementwise.copy()
elementwise_decimal = pandas_kernels.elementwise_decimal.copy()

rowwise = {
    **pandas_kernels.rowwise,
    ops.DateAdd: lambda row: row["left"] + row["right"],
}

reductions = {
    **pandas_kernels.reductions,
    ops.Mode: lambda x: x.mode().loc[0],
    ops.ApproxMedian: lambda x: x.median_approximate(),
    ops.BitAnd: lambda x: x.reduction(np.bitwise_and.reduce),
    ops.BitOr: lambda x: x.reduction(np.bitwise_or.reduce),
    ops.BitXor: lambda x: x.reduction(np.bitwise_xor.reduce),
    # Window functions are calculated locally using pandas
    ops.Last: lambda x: x.compute().iloc[-1] if isinstance(x, dd.Series) else x.iat[-1],
    ops.First: lambda x: x.loc[0] if isinstance(x, dd.Series) else x.iat[0],
}

serieswise = {
    **pandas_kernels.serieswise,
    ops.StringAscii: lambda arg: arg.map(
        ord, na_action="ignore", meta=(arg.name, "int32")
    ),
    ops.TimestampFromUNIX: lambda arg, unit: dd.to_datetime(arg, unit=unit.short),
    ops.DayOfWeekIndex: lambda arg: dd.to_datetime(arg).dt.dayofweek,
    ops.DayOfWeekName: lambda arg: dd.to_datetime(arg).dt.day_name(),
}

# prefer other kernels for the following operations
del generic[ops.IsNull]
del generic[ops.NotNull]
del generic[ops.DateAdd]  # must pass metadata
del serieswise[ops.Round]  # dask series doesn't have a round() method
del serieswise[ops.Strftime]  # doesn't support columnar format strings
del serieswise[ops.Substring]


supported_operations = (
    generic.keys()
    | columnwise.keys()
    | rowwise.keys()
    | serieswise.keys()
    | elementwise.keys()
)
