from __future__ import annotations

import decimal

import dask.dataframe as dd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node


@execute_node.register(ops.Cast, dd.Series, dt.Decimal)
def execute_cast_series_to_decimal(op, data, type, **kwargs):
    precision = type.precision
    scale = type.scale
    context = decimal.Context(prec=precision)
    places = context.create_decimal(f"{'0' * (precision - scale)}.{'0' * scale}")
    return data.apply(
        lambda x, context=context, places=places: (
            context.create_decimal(x).quantize(places)
        ),
        meta=(data.name, "object"),
    )
