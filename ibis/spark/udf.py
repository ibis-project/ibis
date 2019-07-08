"""
APIs for creating user-defined element-wise, reduction and analytic
functions.
"""

import collections
import functools
import itertools

import ibis.expr.datatypes as dt
import ibis.expr.signature as sig
from ibis.pandas.udf import valid_function_signature
from ibis.spark.compiler import SparkUDFNode, compiles

_udf_name_cache = collections.defaultdict(itertools.count)


def create_udf_node(name, fields):
    """Create a new UDF node type.

    Parameters
    ----------
    name : str
        Then name of the UDF node
    fields : OrderedDict
        Mapping of class member name to definition

    Returns
    -------
    result : type
        A new SparkUDFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = '{}_{:d}'.format(name, definition)
    return type(external_name, (SparkUDFNode,), fields)


def udf(input_type, output_type):
    """Define a UDF (user-defined function) that operates element wise on a
    Spark DataFrame.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.spark.udf import udf
    >>> @udf(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(x):
    ...     return x.str.len() * 2
    """
    input_type = list(map(dt.dtype, input_type))
    output_type = dt.dtype(output_type)

    def wrapper(func):
        if not callable(func):
            raise TypeError('func must be callable, got {}'.format(func))

        # validate that the input_type argument and the function signature
        # match
        _ = valid_function_signature(input_type, func)

        # generate a new custom node
        UDFNode = create_udf_node(
            func.__name__,
            {
                'signature': sig.TypeSignature.from_dtypes(input_type),
                'output_type': output_type.column_type,
            },
        )
        # Add func as a property. If added to the class namespace dict, it
        # would be incorrectly used as a bound method, i.e.
        # func(t.col1) would be a call to bound method func with t.col1
        # interpreted as self.
        UDFNode.func = property(lambda self: func)

        @compiles(UDFNode)
        def compiles_udf_node(t, expr):
            return '{}({})'.format(
                UDFNode.__name__,
                ', '.join(map(t.translate, expr.op().args))
            )

        @functools.wraps(func)
        def wrapped(*args):
            return UDFNode(*args).to_expr()

        return wrapped

    return wrapper
