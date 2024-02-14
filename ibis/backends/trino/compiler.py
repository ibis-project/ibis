from __future__ import annotations

import math
from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import (
    FALSE,
    NULL,
    STAR,
    SQLGlotCompiler,
    paren,
)
from ibis.backends.base.sqlglot.datatypes import TrinoType
from ibis.backends.base.sqlglot.dialects import Trino
from ibis.backends.base.sqlglot.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)


class TrinoCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Trino
    type_mapper = TrinoType
    rewrites = (
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        exclude_unsupported_window_frame_from_ops,
        *SQLGlotCompiler.rewrites,
    )
    quoted = True

    NAN = sg.func("nan")
    POS_INF = sg.func("infinity")
    NEG_INF = -POS_INF

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.Quantile,
            ops.MultiQuantile,
            ops.Median,
            ops.RowID,
            ops.TimestampBucket,
        )
    )

    SIMPLE_OPS = {
        ops.Pi: "pi",
        ops.E: "e",
        ops.RegexReplace: "regexp_replace",
        ops.Map: "map",
        ops.MapKeys: "map_keys",
        ops.MapLength: "cardinality",
        ops.MapMerge: "map_concat",
        ops.MapValues: "map_values",
        ops.Log2: "log2",
        ops.Log10: "log10",
        ops.IsNan: "is_nan",
        ops.IsInf: "is_infinite",
        ops.StringToTimestamp: "date_parse",
        ops.Strftime: "date_format",
        ops.ExtractEpochSeconds: "to_unixtime",
        ops.ExtractWeekOfYear: "week_of_year",
        ops.ExtractDayOfYear: "day_of_year",
        ops.ExtractMillisecond: "millisecond",
        ops.ArrayUnion: "array_union",
        ops.ArrayRemove: "array_remove",
        ops.ArrayFlatten: "flatten",
        ops.ArraySort: "array_sort",
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayLength: "cardinality",
        ops.ArrayCollect: "array_agg",
        ops.ArrayIntersect: "array_intersect",
        ops.BitAnd: "bitwise_and_agg",
        ops.BitOr: "bitwise_or_agg",
        ops.TypeOf: "typeof",
        ops.Levenshtein: "levenshtein_distance",
        ops.ExtractProtocol: "url_extract_protocol",
        ops.ExtractHost: "url_extract_host",
        ops.ExtractPath: "url_extract_path",
        ops.ExtractFragment: "url_extract_fragment",
        ops.ArrayPosition: "array_position",
    }

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @staticmethod
    def _minimize_spec(start, end, spec):
        if (
            start is None
            and isinstance(getattr(end, "value", None), ops.Literal)
            and end.value.value == 0
            and end.following
        ):
            return None
        return spec

    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        if op.seed is not None:
            raise com.UnsupportedOperationError(
                "`Table.sample` with a random seed is unsupported"
            )
        sample = sge.TableSample(
            this=parent,
            method="bernoulli" if method == "row" else "system",
            percent=sge.convert(fraction * 100.0),
            seed=None if seed is None else sge.convert(seed),
        )
        return sg.select(STAR).from_(sample)

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                "Trino does not support `sample` correlation"
            )
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    def visit_Arbitrary(self, op, *, arg, how, where):
        if how != "first":
            raise com.UnsupportedOperationError(
                'Trino only supports how="first" for `arbitrary` reduction'
            )
        return self.agg.arbitrary(arg, where=where)

    def visit_BitXor(self, op, *, arg, where):
        a, b = map(sg.to_identifier, "ab")
        input_fn = combine_fn = sge.Lambda(
            this=sge.BitwiseXor(this=a, expression=b), expressions=[a, b]
        )
        return self.agg.reduce_agg(arg, 0, input_fn, combine_fn, where=where)

    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.flatten(self.f.repeat(arg, times))

    def visit_ArraySlice(self, op, *, arg, start, stop):
        def _neg_idx_to_pos(n, idx):
            return self.if_(idx < 0, n + self.f.greatest(idx, -n), idx)

        arg_length = self.f.cardinality(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, _neg_idx_to_pos(arg_length, start))

        if stop is None:
            stop = arg_length
        else:
            stop = _neg_idx_to_pos(arg_length, stop)

        return self.f.slice(arg, start + 1, stop - start)

    def visit_ArrayMap(self, op, *, arg, param, body):
        return self.f.transform(arg, sge.Lambda(this=body, expressions=[param]))

    def visit_ArrayFilter(self, op, *, arg, param, body):
        return self.f.filter(arg, sge.Lambda(this=body, expressions=[param]))

    def visit_ArrayContains(self, op, *, arg, other):
        return self.if_(
            arg.is_(sg.not_(NULL)),
            self.f.coalesce(self.f.contains(arg, other), FALSE),
            NULL,
        )

    def visit_JSONGetItem(self, op, *, arg, index):
        fmt = "%d" if op.index.dtype.is_integer() else '"%s"'
        return self.f.json_extract(arg, self.f.format(f"$[{fmt}]", index))

    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.cast(paren(self.f.day_of_week(arg) + 6) % 7, op.dtype)

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.date_format(arg, "%W")

    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars)

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.substr(arg, -self.f.length(end)).eq(end)

    def visit_Repeat(self, op, *, arg, times):
        return self.f.array_join(self.f.repeat(arg, times), "")

    def visit_DateTimestampTruncate(self, op, *, arg, unit):
        _truncate_precisions = {
            # ms unit is not yet officially documented but it works
            "ms": "millisecond",
            "s": "second",
            "m": "minute",
            "h": "hour",
            "D": "day",
            "W": "week",
            "M": "month",
            "Q": "quarter",
            "Y": "year",
        }

        if (precision := _truncate_precisions.get(unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"Unsupported truncate unit {op.unit!r}"
            )
        return self.f.date_trunc(precision, arg)

    visit_DateTruncate = visit_TimestampTruncate = visit_DateTimestampTruncate

    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.f.from_iso8601_date(
            self.f.format("%04d-%02d-%02d", year, month, day)
        )

    def visit_TimeFromHMS(self, op, *, hours, minutes, seconds):
        return self.cast(
            self.f.format("%02d:%02d:%02d", hours, minutes, seconds), dt.time
        )

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.cast(
            self.f.from_iso8601_timestamp(
                self.f.format(
                    "%04d-%02d-%02dT%02d:%02d:%02d",
                    year,
                    month,
                    day,
                    hours,
                    minutes,
                    seconds,
                )
            ),
            dt.timestamp,
        )

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        short = unit.short
        if short == "ms":
            res = self.f.from_unixtime(self.f.floor(arg / 1_000))
        elif short == "s":
            res = self.f.from_unixtime(arg)
        elif short == "us":
            res = self.f.from_unixtime_nanos((arg - arg % 1_000_000) * 1_000)
        elif short == "ns":
            res = self.f.from_unixtime_nanos(arg - arg % 1_000_000_000)
        else:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported")
        return self.cast(res, op.dtype)

    def visit_StructColumn(self, op, *, names, values):
        return self.cast(sge.Struct(expressions=list(values)), op.dtype)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_floating():
            if math.isfinite(value):
                return self.cast(value, dtype)
            return super().visit_NonNullLiteral(op, value=value, dtype=dtype)
        elif dtype.is_struct():
            items = [
                self.visit_Literal(ops.Literal(v, fdtype), value=v, dtype=fdtype)
                for fdtype, v in zip(dtype.types, value.values())
            ]
            return self.cast(sge.Struct(expressions=items), dtype)
        elif dtype.is_timestamp():
            return self.cast(self.f.from_iso8601_timestamp(value.isoformat()), dtype)
        elif dtype.is_date():
            return self.f.from_iso8601_date(value.isoformat())
        elif dtype.is_time():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_interval():
            return sge.Interval(
                this=sge.convert(str(value)), unit=self.v[dtype.resolution.upper()]
            )
        elif dtype.is_binary():
            return self.f.from_hex(value.hex())
        else:
            return None

    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    def visit_MapGet(self, op, *, arg, key, default):
        return self.f.coalesce(self.f.element_at(arg, key), default)

    def visit_MapContains(self, op, *, arg, key):
        return self.f.contains(self.f.map_keys(arg), key)

    def visit_ExtractFile(self, op, *, arg):
        return self.f.concat_ws(
            "?",
            self.f.nullif(self.f.url_extract_path(arg), ""),
            self.f.nullif(self.f.url_extract_query(arg), ""),
        )

    def visit_ExtractQuery(self, op, *, arg, key):
        if key is None:
            return self.f.url_extract_query(arg)
        return self.f.url_extract_parameter(arg, key)

    def visit_Cot(self, op, *, arg):
        return 1.0 / self.f.tan(arg)

    def visit_StringAscii(self, op, *, arg):
        return self.f.codepoint(
            sge.Cast(
                this=self.f.substr(arg, 1, 2),
                to=sge.DataType(
                    this=sge.DataType.Type.VARCHAR,
                    expressions=[sge.DataTypeParam(this=sge.convert(1))],
                ),
            )
        )

    def visit_ArrayStringJoin(self, op, *, sep, arg):
        return self.f.array_join(arg, sep)

    def visit_First(self, op, *, arg, where):
        return self.f.element_at(self.agg.array_agg(arg, where=where), 1)

    def visit_Last(self, op, *, arg, where):
        return self.f.element_at(self.agg.array_agg(arg, where=where), -1)

    def visit_ArrayZip(self, op, *, arg):
        max_zip_arguments = 5
        chunks = (
            (len(chunk), self.f.zip(*chunk) if len(chunk) > 1 else chunk[0])
            for chunk in toolz.partition_all(max_zip_arguments, arg)
        )

        def combine_zipped(left, right):
            left_n, left_chunk = left
            x, y = map(sg.to_identifier, "xy")

            lhs = list(map(x.__getitem__, range(left_n))) if left_n > 1 else [x]

            right_n, right_chunk = right
            rhs = list(map(y.__getitem__, range(right_n))) if right_n > 1 else [y]

            zipped_chunk = self.f.zip_with(
                left_chunk,
                right_chunk,
                sge.Lambda(this=self.f.row(*lhs, *rhs), expressions=[x, y]),
            )
            return left_n + right_n, zipped_chunk

        all_n, chunk = reduce(combine_zipped, chunks)
        assert all_n == len(op.dtype.value_type)
        return chunk

    def visit_ExtractMicrosecond(self, op, *, arg):
        # trino only seems to store milliseconds, but the result of formatting
        # always pads the right with 000
        return self.cast(self.f.date_format(arg, "%f"), dt.int32)

    def visit_TemporalDelta(self, op, *, part, left, right):
        # trino truncates _after_ the delta, whereas many other backends
        # truncate each operand
        dialect = self.dialect
        return self.f.date_diff(
            part,
            self.f.date_trunc(part, right, dialect=dialect),
            self.f.date_trunc(part, left, dialect=dialect),
            dialect=dialect,
        )

    visit_TimeDelta = visit_DateDelta = visit_TimestampDelta = visit_TemporalDelta

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        unit = op.unit.short
        if unit in ("Y", "Q", "M", "W"):
            raise com.UnsupportedOperationError(f"Interval unit {unit!r} not supported")
        return self.f.parse_duration(
            self.f.concat(
                self.cast(arg, dt.String(nullable=op.arg.dtype.nullable)), unit.lower()
            )
        )

    def visit_Range(self, op, *, start, stop, step):
        def zero_value(dtype):
            if dtype.is_interval():
                # the unit doesn't matter here, because e.g. 0d = 0s
                return self.f.parse_duration("0s")
            return 0

        def interval_sign(v):
            zero = self.f.parse_duration("0s")
            return sge.Case(
                ifs=[
                    self.if_(v.eq(zero), 0),
                    self.if_(v < zero, -1),
                    self.if_(v > zero, 1),
                ]
            )

        def _sign(value, dtype):
            if dtype.is_interval():
                return interval_sign(value)
            return self.f.sign(value)

        step_dtype = op.step.dtype
        zero = zero_value(step_dtype)
        return self.if_(
            sg.and_(
                self.f.nullif(step, zero).is_(sg.not_(NULL)),
                _sign(step, step_dtype).eq(_sign(stop - start, step_dtype)),
            ),
            self.f.array_remove(self.f.sequence(start, stop, step), stop),
            self.f.array(),
        )

    visit_IntegerRange = visit_TimestampRange = visit_Range

    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.element_at(arg, index + 1)

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_integer() and to.is_interval():
            return self.visit_IntervalFromInteger(
                ops.IntervalFromInteger(op.arg, unit=to.unit),
                arg=arg,
                unit=to.unit,
            )
        elif from_.is_integer() and to.is_timestamp():
            return self.f.from_unixtime(arg, to.timezone or "UTC")
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_CountDistinctStar(self, op, *, arg, where):
        make_col = partial(sg.column, table=arg.alias_or_name, quoted=self.quoted)
        row = self.f.row(*map(make_col, op.arg.schema.names))
        return self.agg.count(sge.Distinct(expressions=[row]), where=where)

    def visit_ArrayConcat(self, op, *, arg):
        return self.f.concat(*arg)

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    def visit_RegexpExtract(self, op, *, arg, pattern, index):
        # sqlglot doesn't support the third `group` argument for trino so work
        # around that limitation using an anonymous function
        return self.f.anon.regexp_extract(arg, pattern, index)
