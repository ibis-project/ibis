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
