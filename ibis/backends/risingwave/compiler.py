from __future__ import annotations

from functools import singledispatchmethod

import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.datatypes import RisingWaveType
from ibis.backends.postgres.compiler import PostgresCompiler
from ibis.backends.risingwave.dialect import RisingWave  # noqa: F401


@public
class RisingwaveCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = "risingwave"
    name = "risingwave"
    type_mapper = RisingWaveType

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.name} only implements `pop` correlation coefficient"
            )
        return super().visit_Correlation(
            op, left=left, right=right, how=how, where=where
        )

    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimeTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit):
        unit_mapping = {
            "Y": "year",
            "Q": "quarter",
            "M": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "m": "minute",
            "s": "second",
            "ms": "milliseconds",
            "us": "microseconds",
        }

        if (unit := unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

        return self.f.date_trunc(unit, arg)

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, unit):
        if op.arg.shape == ds.scalar:
            return sge.Interval(this=arg, unit=self.v[unit.name])
        elif op.arg.shape == ds.columnar:
            return arg * sge.Interval(this=sge.convert(1), unit=self.v[unit.name])
        else:
            raise ValueError("Invalid shape for converting to interval")

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        return None

    @visit_node.register(ops.IntegerRange)
    @visit_node.register(ops.TimestampRange)
    @visit_node.register(ops.DateFromYMD)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)
