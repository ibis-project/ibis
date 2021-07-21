"""
Adding and subtracting timestamp/date intervals (dealt with in `_timestamp_op`)
is still WIP since Spark SQL support for these tasks is not comprehensive.
The built-in Spark SQL functions `date_add`, `date_sub`, and `add_months` do
not support timestamps, as they set the HH:MM:SS part to zero. The other option
is arithmetic syntax: <timestamp> + INTERVAL <num> <unit>, where unit is
something like MONTHS or DAYS. However, with the arithmetic syntax, <num>
must be a scalar, i.e. can't be a column like t.int_col.

                        supports        supports        preserves
                        scalars         columns         HH:MM:SS
                     _______________________________ _______________
built-in functions  |               |               |               |
like `date_add`     |      YES      |      YES      |       NO      |
                    |_______________|_______________|_______________|
                    |               |               |               |
arithmetic          |      YES      |       NO      |      YES      |
                    |_______________|_______________|_______________|

"""


import math

import ibis
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sql.compiler import Compiler, ExprTranslator

from .registry import operation_registry

# ----------------------------------------------------------------------
# Select compilation


class SparkUDFNode(ops.ValueOp):
    def output_type(self):
        return rlz.shape_like(self.args, dtype=self.return_type)


class SparkUDAFNode(ops.Reduction):
    def output_type(self):
        return self.return_type.scalar_type()


class SparkExprTranslator(ExprTranslator):
    _registry = operation_registry


rewrites = SparkExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


@rewrites(ops.IsInf)
def spark_rewrites_is_inf(expr):
    arg = expr.op().arg
    return (arg == ibis.literal(math.inf)) | (arg == ibis.literal(-math.inf))


class SparkCompiler(Compiler):
    translator_class = SparkExprTranslator
