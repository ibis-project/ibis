from types import FunctionType, LambdaType

from public import public

from .. import rules as rlz
from ..signature import Argument as Arg
from .analytic import AnalyticOp
from .core import ValueOp, distinct_roots
from .reductions import Reduction


class VectorizedUDF(ValueOp):
    func = Arg(rlz.instance_of((FunctionType, LambdaType)))
    func_args = Arg(rlz.list_of(rlz.column(rlz.any)))
    input_type = Arg(rlz.list_of(rlz.datatype))
    return_type = Arg(rlz.datatype)

    @property
    def inputs(self):
        return self.func_args

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class ElementWiseVectorizedUDF(VectorizedUDF):
    """Node for element wise UDF."""

    def output_type(self):
        return self.return_type.column_type()


@public
class ReductionVectorizedUDF(VectorizedUDF, Reduction):
    """Node for reduction UDF."""

    def output_type(self):
        return self.return_type.scalar_type()


@public
class AnalyticVectorizedUDF(VectorizedUDF, AnalyticOp):
    """Node for analytics UDF."""

    def output_type(self):
        return self.return_type.column_type()
