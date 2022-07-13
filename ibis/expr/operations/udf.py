from types import FunctionType, LambdaType

from ibis.expr import rules as rlz
from ibis.expr.operations.core import Node


class UDFMixin(Node):
    func = rlz.instance_of((FunctionType, LambdaType))
    func_args = rlz.tuple_of(rlz.column(rlz.any))
    # TODO(kszucs): should rename these arguments to
    # input_dtypes and return_dtype
    input_type = rlz.tuple_of(rlz.datatype)
    return_type = rlz.datatype
    func_summary = rlz.optional(rlz.string)
    func_desc = rlz.optional(rlz.string)
