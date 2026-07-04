"""Feldera SQL compiler.

Feldera parses SQL with Apache Calcite using a Postgres-flavoured surface,
so we subclass :class:`PostgresCompiler` and only declare the (small) set of
operations that Feldera/Calcite does not (yet) support.
"""

from __future__ import annotations

from functools import partial

import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.postgres import PostgresCompiler
from ibis.backends.sql.datatypes import FelderaType
from ibis.backends.sql.dialects import Feldera


class FelderaCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = Feldera
    type_mapper = FelderaType

    # Operations we know Feldera/Calcite doesn't support today.  This list is
    # intentionally conservative; we will trim it as the operation matrix is
    # validated against a running pipeline (see `ibis/backends/feldera/tests`).
    UNSUPPORTED_OPS = (
        ops.Sample,
        ops.RandomScalar,
        ops.RandomUUID,
        ops.Arbitrary,
        ops.Mode,
        ops.Kurtosis,
        ops.Quantile,
        ops.MultiQuantile,
        ops.ApproxMultiQuantile,
        ops.First,
        ops.Last,
        ops.RegexSplit,
        ops.TimestampBucket,
        ops.TypeOf,
    )

    # Postgres lowers Sample to TABLESAMPLE; Feldera has no equivalent.
    LOWERED_OPS = {ops.Sample: None}

    def visit_DateFromYMD(self, op, *, year, month, day):
        to_int32 = partial(self.cast, to=dt.int32)
        return sge.Anonymous(
            this="MAKE_DATE",
            expressions=[to_int32(year), to_int32(month), to_int32(day)],
        )

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        # Feldera 0.316 rejects MAKE_TIMESTAMP in ad-hoc queries (lowercases to
        # make_timestamp and fails).  Build an ISO string and parse instead.
        year_str = self.cast(year, dt.string)
        month_str = self.f.lpad(self.cast(month, dt.string), 2, "0")
        day_str = self.f.lpad(self.cast(day, dt.string), 2, "0")
        hours_str = self.f.lpad(self.cast(hours, dt.string), 2, "0")
        minutes_str = self.f.lpad(self.cast(minutes, dt.string), 2, "0")
        seconds_str = self.f.lpad(self.cast(seconds, dt.string), 2, "0")
        ts_str = self.f.concat(
            year_str,
            sge.convert("-"),
            month_str,
            sge.convert("-"),
            day_str,
            sge.convert(" "),
            hours_str,
            sge.convert(":"),
            minutes_str,
            sge.convert(":"),
            seconds_str,
        )
        return self.f.to_timestamp(ts_str)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)

    def visit_StringSplit(self, op, *, arg, delimiter):
        return self.f.split(arg, delimiter)

    def _make_interval(self, arg, unit):
        """Feldera uses ``INTERVAL 'N unit'`` literals, not ``make_interval()``."""
        plural = unit.plural

        if plural == "weeks":
            arg = arg * 7
            plural = "days"

        unit_map = {
            "years": "year",
            "months": "month",
            "days": "day",
            "hours": "hour",
            "minutes": "minute",
            "seconds": "second",
            "milliseconds": "millisecond",
            "microseconds": "microsecond",
            "nanoseconds": "nanosecond",
        }

        unit_str = unit_map.get(plural, plural.rstrip("s"))
        arg_str = self.cast(arg, dt.string)
        interval_str = self.f.concat(arg_str, sge.convert(f" {unit_str}s"))

        return sge.Cast(
            this=interval_str,
            to=sge.DataType(this=sge.DataType.Type.INTERVAL),
        )


compiler = FelderaCompiler()
