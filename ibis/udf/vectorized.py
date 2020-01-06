"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

import ibis.expr.datatypes as dt
from ibis.expr.operations import ElementWiseVectorizedUDF


class UserDefinedFunction(object):
    def __init__(self, func, func_type, input_type, output_type):
        self.func = func
        self.func_type = func_type
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, *args, **kwargs):
        op = self.func_type(
            func=self.func,
            args=args,
            input_type=self.input_type,
            output_type=self.output_type,
        )
        return op.to_expr()


def elementwise(input_type, output_type):
    """ Element wise vectorized UDFs.
    """
    input_type = list(map(dt.dtype, input_type))
    output_type = dt.dtype(output_type)

    def wrapper(func):
        return UserDefinedFunction(
            func, ElementWiseVectorizedUDF, input_type, output_type
        )

    return wrapper
