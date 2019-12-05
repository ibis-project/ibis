"""Top level APIs for defining UDFs that works for multiple backends.

This *currently* mimics ibis.pandas.udf API.

Warning: This is an experimental module and API here can change without notice.
Do not use directly.
"""

import ibis.expr.datatypes as dt
from ibis.expr.operations import ElementWiseUDF


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
    input_type = list(map(dt.dtype, input_type))
    output_type = dt.dtype(output_type)

    def wrapper(func):
        return UserDefinedFunction(
            func, ElementWiseUDF, input_type, output_type
        )

    return wrapper
