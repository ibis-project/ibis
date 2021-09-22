from public import public

from .. import rules as rlz
from ..signature import Argument as Arg
from .analytic import AnalyticOp
from .core import ValueOp, distinct_roots
from .reductions import Reduction


@public
class ElementWiseVectorizedUDF(ValueOp):
    """Node for element wise UDF."""

    func = Arg((types.FunctionType, types.LambdaType))
    func_args = Arg(rlz.list_of(rlz.any))
    input_type = Arg(rlz.list_of(rlz.datatype))
    return_type = Arg(rlz.datatype)

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return rlz.shape_like(self.func_args, dtype=self.return_type)

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class ReductionVectorizedUDF(Reduction):
    """Node for reduction UDF."""

    func = Arg((types.FunctionType, types.LambdaType))
    func_args = Arg(rlz.list_of(rlz.any))
    input_type = Arg(rlz.list_of(rlz.datatype))
    return_type = Arg(rlz.datatype)

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return self.return_type.scalar_type()

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class AnalyticVectorizedUDF(AnalyticOp):
    """Node for analytics UDF."""

    func = Arg((types.FunctionType, types.LambdaType))
    func_args = Arg(rlz.list_of(rlz.any))
    input_type = Arg(rlz.list_of(rlz.datatype))
    return_type = Arg(rlz.datatype)

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return self.return_type.column_type()

    def root_tables(self):
        return distinct_roots(*self.func_args)
