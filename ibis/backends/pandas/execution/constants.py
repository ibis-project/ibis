"""Constants for the pandas backend."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.util

JOIN_TYPES = {
    ops.LeftJoin: "left",
    ops.RightJoin: "right",
    ops.InnerJoin: "inner",
    ops.OuterJoin: "outer",
}


LEFT_JOIN_SUFFIX = f"_ibis_left_{ibis.util.guid()}"
RIGHT_JOIN_SUFFIX = f"_ibis_right_{ibis.util.guid()}"
JOIN_SUFFIXES = LEFT_JOIN_SUFFIX, RIGHT_JOIN_SUFFIX
ALTERNATE_SUFFIXES = {
    LEFT_JOIN_SUFFIX: RIGHT_JOIN_SUFFIX,
    RIGHT_JOIN_SUFFIX: LEFT_JOIN_SUFFIX,
}


IBIS_TYPE_TO_PANDAS_TYPE: dict[dt.DataType, type | str] = {
    dt.float16: np.float16,
    dt.float32: np.float32,
    dt.float64: np.float64,
    dt.float32: np.float32,
    dt.float64: np.float64,
    dt.int8: np.int8,
    dt.int16: np.int16,
    dt.int32: np.int32,
    dt.int64: np.int64,
    dt.string: str,
    dt.timestamp: "datetime64[ns]",
    dt.boolean: np.bool_,
    dt.json: str,
}


IBIS_TO_PYTHON_LITERAL_TYPES = {
    dt.boolean: bool,
    dt.float64: float,
    dt.float32: float,
    dt.int64: int,
    dt.int32: int,
    dt.int16: int,
    dt.int8: int,
    dt.string: str,
    dt.date: lambda x: pd.Timestamp(x).to_pydatetime().date(),
}


BINARY_OPERATIONS = {
    ops.Greater: operator.gt,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.GreaterEqual: operator.ge,
    ops.Equals: operator.eq,
    ops.NotEquals: operator.ne,
    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Modulus: operator.mod,
    ops.Power: operator.pow,
    ops.IdenticalTo: lambda x, y: (x == y) | (pd.isnull(x) & pd.isnull(y)),
    ops.BitwiseXor: lambda x, y: np.bitwise_xor(x, y),
    ops.BitwiseOr: lambda x, y: np.bitwise_or(x, y),
    ops.BitwiseAnd: lambda x, y: np.bitwise_and(x, y),
    ops.BitwiseLeftShift: lambda x, y: np.left_shift(x, y),
    ops.BitwiseRightShift: lambda x, y: np.right_shift(x, y),
}
