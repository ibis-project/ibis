"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import numpy as np
import sqlglot as sg
from multipledispatch import Dispatcher

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.registry import (
    fixed_arity,
    helpers,
    operation_registry,
    reduction,
    unary,
)
from ibis.backends.base.sql.registry.literal import _string_literal_format
from ibis.backends.base.sql.registry.main import table_array_view
from ibis.backends.bigquery.datatypes import BigQueryType
from ibis.common.temporal import DateUnit, IntervalUnit, TimeUnit

if TYPE_CHECKING:
    from ibis.backends.base.sql import compiler


# def _regex_extract(translator, op):
#     matches = f"REGEXP_CONTAINS({arg}, {regex})"
#     # non-greedily match the regex's prefix so the regex can match as much as possible
#     nonzero_index_replace = rf"REGEXP_REPLACE({arg}, CONCAT('.*?', {regex}, '.*'), CONCAT('\\', CAST({index} AS STRING)))"
#     # zero index replacement means capture everything matched by the regex, so
#     # we wrap the regex in an outer group
#     zero_index_replace = (
#         rf"REGEXP_REPLACE({arg}, CONCAT('.*?', CONCAT('(', {regex}, ')'), '.*'), '\\1')"
#     )
#     extract = f"IF({index} = 0, {zero_index_replace}, {nonzero_index_replace})"
#     return f"IF({matches}, {extract}, NULL)"


def _date_binary(func):
    def _formatter(translator, op):
        arg, offset = op.left, op.right

        unit = offset.dtype.unit
        if not unit.is_date():
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
            )

        formatted_arg = translator.translate(arg)
        formatted_offset = translator.translate(offset)
        return f"{func}({formatted_arg}, {formatted_offset})"

    return _formatter


def _timestamp_binary(func):
    def _formatter(translator, op):
        arg, offset = op.left, op.right

        unit = offset.dtype.unit
        if unit == IntervalUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
            )

        if unit.is_date():
            try:
                offset = offset.to_expr().to_unit("h").op()
            except ValueError:
                raise com.UnsupportedOperationError(
                    f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
                )

        formatted_arg = translator.translate(arg)
        formatted_offset = translator.translate(offset)
        return f"{func}({formatted_arg}, {formatted_offset})"

    return _formatter


OPERATION_REGISTRY = {
    **operation_registry,
    # Math
    # Temporal functions
    # ops.RegexExtract: _regex_extract,
    # ops.Cast: _cast,
}
