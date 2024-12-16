from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers import PostgresCompiler
from ibis.backends.sql.compilers.base import ALL_OPERATIONS, NULL
from ibis.backends.sql.datatypes import RisingWaveType
from ibis.backends.sql.dialects import RisingWave


class RisingWaveCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = RisingWave
    type_mapper = RisingWaveType

    LOWERED_OPS = {ops.Sample: None}

    UNSUPPORTED_OPS = (
        ops.Arbitrary,
        ops.Mode,
        ops.RandomScalar,
        ops.RandomUUID,
        ops.MultiQuantile,
        ops.ApproxMultiQuantile,
        ops.Sample,
        *(
            op
            for op in ALL_OPERATIONS
            if issubclass(
                op, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp, ops.GeoUnaryUnion)
            )
        ),
    )

    def visit_DateNow(self, op):
        return self.cast(sge.CurrentTimestamp(), dt.date)

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if not order_by:
            raise com.UnsupportedOperationError(
                "RisingWave requires an `order_by` be specified in `first`"
            )
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.first_value(arg, where=where, order_by=order_by)

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if not order_by:
            raise com.UnsupportedOperationError(
                "RisingWave requires an `order_by` be specified in `last`"
            )
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.last_value(arg, where=where, order_by=order_by)

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                "RisingWave only implements `pop` correlation coefficient"
            )
        return super().visit_Correlation(
            op, left=left, right=right, how=how, where=where
        )

    def visit_Quantile(self, op, *, arg, quantile, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        return sge.WithinGroup(
            this=self.f[f"percentile_{suffix}"](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )

    visit_ApproxQuantile = visit_Quantile

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

    visit_TimeTruncate = visit_DateTruncate = visit_TimestampTruncate

    def _make_interval(self, arg, unit):
        return arg * sge.Interval(this=sge.convert(1), unit=self.v[unit.name])

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
            return self.cast("".join(map(r"\x{:0>2x}".format, value)), dt.binary)
        elif dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_json():
            return sge.convert(str(value))
        return None


compiler = RisingWaveCompiler()
