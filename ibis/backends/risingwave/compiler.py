from __future__ import annotations

import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import ALL_OPERATIONS
from ibis.backends.base.sqlglot.datatypes import RisingWaveType
from ibis.backends.base.sqlglot.dialects import RisingWave
from ibis.backends.postgres.compiler import PostgresCompiler


@public
class RisingwaveCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = RisingWave
    type_mapper = RisingWaveType

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.DateFromYMD,
            ops.Mode,
            *(
                op
                for op in ALL_OPERATIONS
                if issubclass(
                    op, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp, ops.GeoUnaryUnion)
                )
            ),
        )
    )

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                "RisingWave only implements `pop` correlation coefficient"
            )
        return super().visit_Correlation(
            op, left=left, right=right, how=how, where=where
        )

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

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        if op.arg.shape == ds.scalar:
            return sge.Interval(this=arg, unit=self.v[unit.name])
        elif op.arg.shape == ds.columnar:
            return arg * sge.Interval(this=sge.convert(1), unit=self.v[unit.name])
        else:
            raise ValueError("Invalid shape for converting to interval")

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
            return self.cast("".join(map(r"\x{:0>2x}".format, value)), dt.binary)
        elif dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_json():
            return sge.convert(str(value))
        return None


_SIMPLE_OPS = {
    ops.First: "first_value",
    ops.Last: "last_value",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(RisingwaveCompiler, f"visit_{_op.__name__}", _fmt)
