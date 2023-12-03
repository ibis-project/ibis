from __future__ import annotations


import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute


@execute.register(ops.ElementWiseVectorizedUDF)
def execute_elementwise_udf(op, func, func_args, input_type, return_type):
    """Execute an elementwise UDF."""

    res = func(*func_args)
    if isinstance(res, pd.DataFrame):
        # it is important otherwise it is going to fill up the memory
        res = res.apply(lambda row: row.to_dict(), axis=1)

    return res


@execute.register(ops.ReductionVectorizedUDF)
def execute_reduction_udf(op, func, func_args, input_type, return_type):
    """Execute a reduction UDF."""

    def agg(df):
        args = [df[col] for col in func_args]
        return func(*args)

    return agg
