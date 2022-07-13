from public import public

from ibis.expr import rules as rlz
from ibis.expr.operations.analytic import AnalyticOp
from ibis.expr.operations.core import ValueOp, distinct_roots
from ibis.expr.operations.reductions import Reduction
from ibis.expr.operations.udf import UDFMixin


class VectorizedUDF(ValueOp, UDFMixin):
    @property
    def inputs(self):
        return self.func_args

    @property
    def output_dtype(self):
        return self.return_type

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class ElementWiseVectorizedUDF(VectorizedUDF):
    """Node for element wise UDF."""

    output_shape = rlz.Shape.COLUMNAR


@public
class ReductionVectorizedUDF(VectorizedUDF, Reduction):
    """Node for reduction UDF."""

    output_shape = rlz.Shape.SCALAR


# TODO(kszucs): revisit
@public
class AnalyticVectorizedUDF(VectorizedUDF, AnalyticOp):
    """Node for analytics UDF."""

    output_shape = rlz.Shape.COLUMNAR
