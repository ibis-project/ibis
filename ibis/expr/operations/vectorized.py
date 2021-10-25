from public import public

from .. import rules as rlz
from ..signature import Argument as Arg
from .analytic import AnalyticOp
from .core import ValueOp, distinct_roots
from .reductions import Reduction


@public
class ElementWiseVectorizedUDF(ValueOp):
    """Node for element wise UDF."""

    func = Arg(callable)
    func_args = Arg(tuple)
    input_type = Arg(rlz.shape_like('func_args'))
    _output_type = Arg(rlz.noop)

    def __init__(self, func, args, input_type, output_type):
        self.func = func
        self.func_args = args
        self.input_type = input_type
        self._output_type = output_type

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return self._output_type.column_type()

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class ReductionVectorizedUDF(Reduction):
    """Node for reduction UDF."""

    func = Arg(callable)
    func_args = Arg(tuple)
    input_type = Arg(rlz.shape_like('func_args'))
    _output_type = Arg(rlz.noop)

    def __init__(self, func, args, input_type, output_type):
        self.func = func
        self.func_args = args
        self.input_type = input_type
        self._output_type = output_type

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return self._output_type.scalar_type()

    def root_tables(self):
        return distinct_roots(*self.func_args)


@public
class AnalyticVectorizedUDF(AnalyticOp):
    """Node for analytics UDF."""

    func = Arg(callable)
    func_args = Arg(tuple)
    input_type = Arg(rlz.shape_like('func_args'))
    _output_type = Arg(rlz.noop)

    def __init__(self, func, args, input_type, output_type):
        self.func = func
        self.func_args = args
        self.input_type = input_type
        self._output_type = output_type

    @property
    def inputs(self):
        return self.func_args

    def output_type(self):
        return self._output_type.column_type()

    def root_tables(self):
        return distinct_roots(*self.func_args)
