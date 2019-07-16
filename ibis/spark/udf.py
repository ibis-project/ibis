"""
APIs for creating user-defined element-wise, reduction and analytic
functions.
"""

import collections
import functools
import itertools

import pyspark.sql.types as pt
from multipledispatch import Dispatcher
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.functions import udf as pyspark_udf

import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.signature as sig
from ibis.pandas.udf import valid_function_signature
from ibis.spark.client import _SPARK_DTYPE_TO_IBIS_DTYPE
from ibis.spark.compiler import SparkUDAFNode, SparkUDFNode, compiles

_udf_name_cache = collections.defaultdict(itertools.count)

_IBIS_DTYPE_TO_SPARK_DTYPE = dict(
    reversed(t) for t in _SPARK_DTYPE_TO_IBIS_DTYPE.items()
)

spark_dtype = Dispatcher('spark_dtype')


@spark_dtype.register(object)
def default(value, **kwargs) -> pt.DataType:
    raise com.IbisTypeError('Value {!r} is not a valid datatype'.format(value))


@spark_dtype.register(pt.DataType)
def from_spark_dtype(value: pt.DataType) -> pt.DataType:
    return value


@spark_dtype.register(dt.DataType)
def ibis_dtype_to_spark_dtype(ibis_dtype_obj):
    """Convert ibis types types to Spark SQL."""
    return _IBIS_DTYPE_TO_SPARK_DTYPE.get(type(ibis_dtype_obj))()


@spark_dtype.register(dt.Decimal)
def ibis_decimal_dtype_to_spark_dtype(ibis_dtype_obj):
    precision = ibis_dtype_obj.precision
    scale = ibis_dtype_obj.scale
    return pt.DecimalType(precision, scale)


@spark_dtype.register(dt.Array)
def ibis_array_dtype_to_spark_dtype(ibis_dtype_obj):
    element_type = spark_dtype(ibis_dtype_obj.value_type)
    contains_null = ibis_dtype_obj.value_type.nullable
    return pt.ArrayType(element_type, contains_null)


@spark_dtype.register(dt.Map)
def ibis_map_dtype_to_spark_dtype(ibis_dtype_obj):
    key_type = spark_dtype(ibis_dtype_obj.key_type)
    value_type = spark_dtype(ibis_dtype_obj.value_type)
    value_contains_null = ibis_dtype_obj.value_type.nullable
    return pt.MapType(key_type, value_type, value_contains_null)


@spark_dtype.register(dt.Struct)
def ibis_struct_dtype_to_spark_dtype(ibis_dtype_obj):
    names = ibis_dtype_obj.names
    spark_types = list(map(spark_dtype, ibis_dtype_obj.types))
    fields = [
        pt.StructField(n, t, t.nullable)
        for (n, t) in zip(names, spark_types)
    ]
    return pt.StructType(fields)


def validate_func_and_types(input_type, output_type, func):
    _input_type = list(map(dt.dtype, input_type))
    _output_type = dt.dtype(output_type)

    if not callable(func):
        raise TypeError('func must be callable, got {}'.format(func))

    # validate that the input_type argument and the function signature
    # match
    _ = valid_function_signature(_input_type, func)

    if not output_type.nullable:
        raise com.IbisTypeError(
            'Spark does not support non-nullable output types'
        )

    return _input_type, _output_type


def create_udf_node(name, fields, udf_func, base_class):
    """Create a new UDF node type.

    Parameters
    ----------
    name : str
        Then name of the UDF node
    fields : OrderedDict
        Mapping of class member name to definition
    udf_func : function
        Should be the result of calling pyspark.sql.functions.udf or
        pyspark.sql.functions.pandas_udf on the user-specified func
    base_class : Union[SparkUDFNode, SparkUDAFNode]

    Returns
    -------
    result : type
        A new SparkUDFNode or SparkUDAFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = '{}_{:d}'.format(name, definition)
    UDFNode = type(external_name, (base_class,), fields)
    # Add udf_func as a property. If added to the class namespace dict, it
    # would be incorrectly used as a bound method, i.e.
    # udf_func(t.column) would be a call to bound method func with t.column
    # interpreted as self.
    UDFNode.udf_func = property(lambda self: udf_func)
    return UDFNode


class udf:
    @staticmethod
    def elementwise(input_type, output_type):
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
        >>> @udf.elementwise(input_type=[dt.string], output_type=dt.int64)
        ... def my_string_length(x):
        ...     return x.str.len() * 2
        """
        def wrapper(func):
            _input_type, _output_type = validate_func_and_types(
                input_type, output_type, func
            )

            spark_output_type = spark_dtype(_output_type)
            udf_func = pyspark_udf(func, spark_output_type)

            # generate a new custom node
            UDFNode = create_udf_node(
                udf_func.__name__,
                {
                    'signature': sig.TypeSignature.from_dtypes(_input_type),
                    'output_type': _output_type.column_type,
                },
                udf_func,
                SparkUDFNode
            )

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

    @staticmethod
    def elementwise_pandas(input_type, output_type):
        """Define a Pandas UDF (user-defined function) that operates element wise on a
        Spark DataFrame.
        """
        def wrapper(func):
            _input_type, _output_type = validate_func_and_types(
                input_type, output_type, func
            )

            spark_output_type = spark_dtype(_output_type)
            if isinstance(spark_output_type, (pt.MapType, pt.StructType)):
                raise com.IbisTypeError(
                    'Spark does not support MapType or StructType output \
                        for Pandas UDFs'
                )
            udf_func = pandas_udf(
                func, spark_output_type, PandasUDFType.SCALAR
            )

            # generate a new custom node
            UDFNode = create_udf_node(
                func.__name__,
                {
                    'signature': sig.TypeSignature.from_dtypes(_input_type),
                    'output_type': _output_type.column_type,
                },
                udf_func,
                SparkUDFNode
            )

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

    @staticmethod
    def reduction(input_type, output_type):
        """Define a user-defined reduction function that takes N pandas Series
        or scalar values as inputs and produces one row of output.

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
        >>> @udf.reduction(input_type=[dt.string], output_type=dt.int64)
        ... def my_string_length_agg(series, **kwargs):
        ...     return (series.str.len() * 2).sum()
        """
        def wrapper(func):
            _input_type, _output_type = validate_func_and_types(
                input_type, output_type, func
            )

            spark_output_type = spark_dtype(_output_type)
            if isinstance(spark_output_type, (pt.MapType, pt.StructType)):
                raise com.IbisTypeError(
                    'Spark does not support MapType or StructType output \
                        for Pandas UDFs'
                )
            udf_func = pandas_udf(
                func, spark_output_type, PandasUDFType.GROUPED_AGG
            )

            # generate a new custom node
            UDFNode = create_udf_node(
                func.__name__,
                {
                    'signature': sig.TypeSignature.from_dtypes(_input_type),
                    'output_type': _output_type.scalar_type,
                },
                udf_func,
                SparkUDAFNode
            )

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
