from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers import PostgresCompiler
from ibis.backends.sql.compilers.base import ALL_OPERATIONS, NULL, SQLGlotCompiler
from ibis.backends.sql.datatypes import RisingWaveType
from ibis.backends.sql.dialects import RisingWave

if TYPE_CHECKING:
    from collections.abc import Mapping

    import ibis.expr.types as ir


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
        ops.Kurtosis,
        *(
            op
            for op in ALL_OPERATIONS
            if issubclass(
                op, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp, ops.GeoUnaryUnion)
            )
        ),
    )

    SIMPLE_OPS = PostgresCompiler.SIMPLE_OPS | {
        ops.MapLength: "map_length",
        ops.Map: "map_from_key_values",
        ops.MapKeys: "map_keys",
        ops.MapValues: "map_values",
    }

    del SIMPLE_OPS[ops.MapContains]

    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ):
        table_expr = expr.as_table()
        schema = table_expr.schema()

        conversions = {name: table_expr[name].as_ewkb() for name in schema.geospatial}
        conversions.update(
            (col, table_expr[col].cast(dt.string))
            for col, typ in schema.items()
            if typ.is_json()
        )
        # convert maps to json, otherwise the output format is a custom
        # risingwave string syntax
        conversions.update(
            (col, table_expr[col].cast(dt.JSON(binary=True)))
            for col, typ in schema.items()
            if typ.is_map()
        )

        if conversions:
            table_expr = table_expr.mutate(**conversions)
        return SQLGlotCompiler.to_sqlglot(self, table_expr, limit=limit, params=params)

    def visit_DateNow(self, op):
        return self.cast(sge.CurrentTimestamp(), dt.date)

    def visit_Cast(self, op, *, arg, to):
        if to.is_json():
            return self.f.to_jsonb(arg)
        else:
            return super().visit_Cast(op, arg=arg, to=to)

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
        elif dtype.is_map():
            return self.f.map_from_key_values(
                self.f.array(*value.keys()), self.f.array(*value.values())
            )
        return None

    def visit_MapGet(self, op, *, arg, key, default):
        if op.dtype.is_null():
            return NULL
        else:
            arg_dtype = op.arg.dtype
            key_type = arg_dtype.key_type
            value_type = dt.higher_precedence(arg_dtype.value_type, op.default.dtype)
            new_dtype = dt.Map(key_type, value_type, nullable=arg_dtype.nullable)
            return self.f.coalesce(
                self.f.map_access(
                    self.cast(arg, dt.higher_precedence(new_dtype, arg_dtype)),
                    self.cast(key, key_type),
                ),
                default,
            )

    def visit_MapMerge(self, op, *, left, right):
        return self.if_(
            left.is_(NULL).or_(right.is_(NULL)), NULL, self.f.map_cat(left, right)
        )

    def visit_MapContains(self, op, *, arg, key):
        return self.f.map_contains(
            self.cast(arg, op.arg.dtype), self.cast(key, op.key.dtype)
        )


compiler = RisingWaveCompiler()
