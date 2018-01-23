"""Constants for the pandas backend.
"""

import operator

import numpy as np
import pandas as pd

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


JOIN_TYPES = {
    ops.LeftJoin: 'left',
    ops.RightJoin: 'right',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


LEFT_JOIN_SUFFIX = '_ibis_left_{}'.format(ibis.util.guid())
RIGHT_JOIN_SUFFIX = '_ibis_right_{}'.format(ibis.util.guid())
JOIN_SUFFIXES = LEFT_JOIN_SUFFIX, RIGHT_JOIN_SUFFIX
ALTERNATE_SUFFIXES = {
    LEFT_JOIN_SUFFIX: RIGHT_JOIN_SUFFIX,
    RIGHT_JOIN_SUFFIX: LEFT_JOIN_SUFFIX,
}


IBIS_TYPE_TO_PANDAS_TYPE = {
    dt.float: np.float32,
    dt.double: np.float64,
    dt.int8: np.int8,
    dt.int16: np.int16,
    dt.int32: np.int32,
    dt.int64: np.int64,
    dt.string: str,
    dt.timestamp: 'datetime64[ns]',
}


IBIS_TO_PYTHON_LITERAL_TYPES = {
    dt.double: float,
    dt.float: float,
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

    ops.StringConcat: operator.add,
}
