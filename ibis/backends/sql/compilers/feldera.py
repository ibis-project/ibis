"""Feldera SQL compiler.

Ad-hoc SQL (what ``Backend.execute()`` runs via ``Pipeline.query_arrow``) is
parsed by Apache DataFusion, so we subclass :class:`PostgresCompiler` and only
override the operations whose Postgres lowering DataFusion rejects.  See
``ibis/backends/feldera/tests/test_adhoc_surface.py`` for the validated
surface.
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

    # Operations we know Feldera ad-hoc (DataFusion) does not support today.
    # This list is intentionally conservative; trim it as the operation matrix
    # is validated against a running pipeline (see test_adhoc_surface.py).
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

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        # DataFusion ad-hoc rejects MAKE_TIMESTAMP ("Invalid function
        # 'make_timestamp'").  Build an ISO 8601 string and parse it with
        # TO_TIMESTAMP, which is accepted.
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

    def visit_Clip(self, op, *, arg, lower, upper):
        # The base compiler wraps GREATEST/LEAST in ``CASE WHEN arg IS NULL
        # THEN arg ELSE ... END`` to propagate NULLs.  DataFusion's optimizer
        # miscompiles the ``THEN arg`` variant (returns 0 for ~55 % of rows
        # that are not NULL), so use ``THEN NULL`` instead, which is
        # semantically identical (``arg`` *is* NULL in that branch).
        if upper is not None:
            arg = self.if_(arg.is_(sge.Null()), sge.Null(), self.f.least(upper, arg))
        if lower is not None:
            arg = self.if_(arg.is_(sge.Null()), sge.Null(), self.f.greatest(lower, arg))
        return arg

    def visit_Log2(self, op, *, arg):
        # Postgres casts to DECIMAL before calling LOG(base, arg), but Feldera's
        # DECIMAL without an explicit scale truncates fractional digits, so
        # LOG(CAST(2 AS DECIMAL), CAST(11.1 AS DECIMAL)) returns 3.0 instead of
        # 3.472.  DataFusion ad-hoc has LOG2, so use it directly.
        return self.f.log2(arg)

    def visit_Log(self, op, *, arg, base):
        # Same DECIMAL-truncation issue as visit_Log2: emit LOG(base, arg)
        # without casting through DECIMAL.  When base is None, emit LN(arg).
        if base is None:
            return self.f.ln(arg)
        return self.f.log(base, arg)

    def _make_interval(self, arg, unit):
        """Lower ``IntervalFromInteger`` without ``make_interval()``.

        DataFusion ad-hoc has no ``make_interval``; instead, cast the integer
        to a string and parse it as an interval literal
        (``CAST('3 days' AS INTERVAL)``).
        """
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
