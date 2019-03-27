from __future__ import absolute_import

import decimal
import math
import numbers

import numpy as np
import pandas as pd
import six

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.pandas.dispatch import execute_node


@execute_node.register(ops.Ln, decimal.Decimal)
def execute_decimal_natural_log(op, data, **kwargs):
    try:
        return data.ln()
    except decimal.InvalidOperation:
        return decimal.Decimal('NaN')


@execute_node.register(ops.Log, decimal.Decimal, decimal.Decimal)
def execute_decimal_log_with_decimal_base(op, data, base, **kwargs):
    try:
        return data.ln() / base.ln()
    except decimal.InvalidOperation:
        return decimal.Decimal('NaN')


@execute_node.register(ops.Log, decimal.Decimal, type(None))
def execute_decimal_log_with_no_base(op, data, _, **kwargs):
    return execute_decimal_natural_log(op, data, **kwargs)


@execute_node.register(ops.Log, decimal.Decimal, numbers.Real)
def execute_decimal_log_with_real_base(op, data, base, **kwargs):
    return execute_node(op, data, decimal.Decimal(base), **kwargs)


@execute_node.register(ops.Log, decimal.Decimal, np.integer)
def execute_decimal_log_with_np_integer_base(op, data, base, **kwargs):
    return execute_node(op, data, int(base), **kwargs)


@execute_node.register(ops.Log2, decimal.Decimal)
def execute_decimal_log2(op, data, **kwargs):
    try:
        return data.ln() / decimal.Decimal(2).ln()
    except decimal.InvalidOperation:
        return decimal.Decimal('NaN')


# While ops.Negate is a subclass of ops.UnaryOp, multipledispatch will be
# faster if we provide types that can potentially match the types of inputs
# exactly
@execute_node.register((ops.UnaryOp, ops.Negate), decimal.Decimal)
def execute_decimal_unary(op, data, **kwargs):
    operation_name = type(op).__name__.lower()
    math_function = getattr(math, operation_name, None)
    function = getattr(
        decimal.Decimal,
        operation_name,
        lambda x: decimal.Decimal(math_function(x))
    )
    try:
        return function(data)
    except decimal.InvalidOperation:
        return decimal.Decimal('NaN')


@execute_node.register(ops.Sign, decimal.Decimal)
def execute_decimal_sign(op, data, **kwargs):
    return data if not data else decimal.Decimal(1).copy_sign(data)


@execute_node.register(ops.Abs, decimal.Decimal)
def execute_decimal_abs(op, data, **kwargs):
    return abs(data)


@execute_node.register(
    ops.Round, decimal.Decimal, (np.integer,) + six.integer_types
)
def execute_round_decimal(op, data, places, **kwargs):
    # If we only allowed Python 3, we wouldn't have to implement any of this;
    # we could just call round(data, places) :(
    tuple_value = data.as_tuple()
    precision = len(tuple_value.digits)
    integer_part_length = precision + min(tuple_value.exponent, 0)

    if places < 0:
        decimal_format_string = '0.{}E+{:d}'.format(
            '0' * (integer_part_length - 1 + places),
            max(integer_part_length + places, abs(places))
        )
    else:
        decimal_format_string = '{}.{}'.format(
            '0' * integer_part_length, '0' * places
        )

    places = decimal.Decimal(decimal_format_string)
    return data.quantize(places)


@execute_node.register(ops.Round, decimal.Decimal, type(None))
def execute_round_decimal_no_places(op, data, _, **kwargs):
    return np.int64(round(data))


@execute_node.register(ops.Cast, pd.Series, dt.Decimal)
def execute_cast_series_to_decimal(op, data, type, **kwargs):
    precision = type.precision
    scale = type.scale
    context = decimal.Context(prec=precision)
    places = context.create_decimal(
        '{}.{}'.format('0' * (precision - scale), '0' * scale),
    )
    return data.apply(
        lambda x, context=context, places=places: (  # noqa: E501
            context.create_decimal(x).quantize(places)
        )
    )
