from types import FunctionType, LambdaType

from public import public

from ibis.expr import rules as rlz
from ibis.expr.operations.analytic import Analytic
from ibis.expr.operations.core import Value, distinct_roots
from ibis.expr.operations.reductions import Reduction


class VectorizedUDF(Value):
    func = rlz.instance_of((FunctionType, LambdaType))
    func_args = rlz.tuple_of(rlz.column(rlz.any))
    # TODO(kszucs): should rename these arguments to
    # input_dtypes and return_dtype
    input_type = rlz.tuple_of(rlz.datatype)
    return_type = rlz.datatype

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
class AnalyticVectorizedUDF(VectorizedUDF, Analytic):
    """Node for analytics UDF."""

    output_shape = rlz.Shape.COLUMNAR
