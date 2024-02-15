from __future__ import annotations

import math
from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.dialects import DuckDB

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import (
    NULL,
    STAR,
    SQLGlotCompiler,
)
from ibis.backends.base.sqlglot.datatypes import DuckDBType

_INTERVAL_SUFFIXES = {
    "ms": "milliseconds",
    "us": "microseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "D": "days",
    "M": "months",
    "Y": "years",
}


@public
class DuckDBCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = DuckDB
    type_mapper = DuckDBType

    SIMPLE_OPS = {
        ops.ArrayPosition: "list_indexof",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.EndsWith: "suffix",
        ops.Hash: "hash",
        ops.IntegerRange: "range",
        ops.TimestampRange: "range",
        ops.MapKeys: "map_keys",
        ops.MapLength: "cardinality",
        ops.MapMerge: "map_concat",
        ops.MapValues: "map_values",
        ops.Mode: "mode",
        ops.TimeFromHMS: "make_time",
        ops.TypeOf: "typeof",
        ops.GeoPoint: "st_point",
        ops.GeoAsText: "st_astext",
        ops.GeoArea: "st_area",
        ops.GeoBuffer: "st_buffer",
        ops.GeoCentroid: "st_centroid",
        ops.GeoContains: "st_contains",
        ops.GeoCovers: "st_covers",
        ops.GeoCoveredBy: "st_coveredby",
        ops.GeoCrosses: "st_crosses",
        ops.GeoDifference: "st_difference",
        ops.GeoDisjoint: "st_disjoint",
        ops.GeoDistance: "st_distance",
        ops.GeoDWithin: "st_dwithin",
        ops.GeoEndPoint: "st_endpoint",
        ops.GeoEnvelope: "st_envelope",
        ops.GeoEquals: "st_equals",
        ops.GeoFlipCoordinates: "st_flipcoordinates",
        ops.GeoGeometryType: "st_geometrytype",
        ops.GeoIntersection: "st_intersection",
        ops.GeoIntersects: "st_intersects",
        ops.GeoIsValid: "st_isvalid",
        ops.GeoLength: "st_length",
        ops.GeoNPoints: "st_npoints",
        ops.GeoOverlaps: "st_overlaps",
        ops.GeoStartPoint: "st_startpoint",
        ops.GeoTouches: "st_touches",
        ops.GeoUnion: "st_union",
        ops.GeoUnaryUnion: "st_union_agg",
        ops.GeoWithin: "st_within",
        ops.GeoX: "st_x",
        ops.GeoY: "st_y",
    }

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    def visit_StructColumn(self, op, *, names, values):
        return sge.Struct.from_arg_list(
            [
                sge.PropertyEQ(
                    this=sg.to_identifier(name, quoted=self.quoted), expression=value
                )
                for name, value in zip(names, values)
            ]
        )

    def visit_ArrayDistinct(self, op, *, arg):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.list_extract(arg, index + self.cast(index >= 0, op.index.dtype))

    def visit_ArrayRepeat(self, op, *, arg, times):
        func = sge.Lambda(this=arg, expressions=[sg.to_identifier("_")])
        return self.f.flatten(self.f.list_apply(self.f.range(times), func))

    # TODO(kszucs): this could be moved to the base SQLGlotCompiler
    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        sample = sge.TableSample(
            this=parent,
            method="bernoulli" if method == "row" else "system",
            percent=sge.convert(fraction * 100.0),
            seed=None if seed is None else sge.convert(seed),
        )
        return sg.select(STAR).from_(sample)

    def visit_ArraySlice(self, op, *, arg, start, stop):
        arg_length = self.f.len(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, self._neg_idx_to_pos(arg, start))

        if stop is None:
            stop = arg_length
        else:
            stop = self._neg_idx_to_pos(arg, stop)

        return self.f.list_slice(arg, start + 1, stop)

    def visit_ArrayMap(self, op, *, arg, body, param):
        lamduh = sge.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_apply(arg, lamduh)

    def visit_ArrayFilter(self, op, *, arg, body, param):
        lamduh = sge.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_filter(arg, lamduh)

    def visit_ArrayIntersect(self, op, *, left, right):
        param = sg.to_identifier("x")
        body = self.f.list_contains(right, param)
        lamduh = sge.Lambda(this=body, expressions=[param])
        return self.f.list_filter(left, lamduh)

    def visit_ArrayRemove(self, op, *, arg, other):
        param = sg.to_identifier("x")
        body = param.neq(other)
        lamduh = sge.Lambda(this=body, expressions=[param])
        return self.f.list_filter(arg, lamduh)

    def visit_ArrayUnion(self, op, *, left, right):
        arg = self.f.list_concat(left, right)
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    def visit_ArrayZip(self, op, *, arg):
        i = sg.to_identifier("i")
        body = sge.Struct.from_arg_list(
            [
                sge.Slice(this=k, expression=v[i])
                for k, v in zip(map(sge.convert, op.dtype.value_type.names), arg)
            ]
        )
        func = sge.Lambda(this=body, expressions=[i])
        return self.f.list_apply(
            self.f.range(
                1,
                # DuckDB Range excludes upper bound
                self.f.greatest(*map(self.f.len, arg)) + 1,
            ),
            func,
        )

    def visit_MapGet(self, op, *, arg, key, default):
        return self.f.ifnull(
            self.f.list_extract(self.f.element_at(arg, key), 1), default
        )

    def visit_MapContains(self, op, *, arg, key):
        return self.f.len(self.f.element_at(arg, key)).neq(0)

    def visit_ToJSONMap(self, op, *, arg):
        return sge.TryCast(this=arg, to=self.type_mapper.from_ibis(op.dtype))

    def visit_ToJSONArray(self, op, *, arg):
        return self.visit_ToJSONMap(op, arg=arg)

    def visit_ArrayConcat(self, op, *, arg):
        # TODO(cpcloud): map ArrayConcat to this in sqlglot instead of here
        return reduce(self.f.list_concat, arg)

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        if unit.short == "ns":
            raise com.UnsupportedOperationError(
                f"{self.dialect} doesn't support nanosecond interval resolutions"
            )

        if unit.singular == "week":
            return self.f.to_days(arg * 7)
        return self.f[f"to_{unit.plural}"](arg)

    def visit_FindInSet(self, op, *, needle, values):
        return self.f.list_indexof(self.f.array(*values), needle)

    def visit_CountDistinctStar(self, op, *, where, arg):
        # use a tuple because duckdb doesn't accept COUNT(DISTINCT a, b, c, ...)
        #
        # this turns the expression into COUNT(DISTINCT (a, b, c, ...))
        row = sge.Tuple(
            expressions=list(
                map(partial(sg.column, quoted=self.quoted), op.arg.schema.keys())
            )
        )
        return self.agg.count(sge.Distinct(expressions=[row]), where=where)

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.f.mod(self.f.extract("ms", arg), 1_000)

    # DuckDB extracts subminute microseconds and milliseconds
    # so we have to finesse it a little bit
    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.mod(self.f.extract("us", arg), 1_000_000)

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        unit = unit.short
        if unit == "ms":
            return self.f.epoch_ms(arg)
        elif unit == "s":
            return sge.UnixToTime(this=arg)
        else:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds, **_
    ):
        args = [year, month, day, hours, minutes, seconds]

        func = "make_timestamp"
        if (timezone := op.dtype.timezone) is not None:
            func += "tz"
            args.append(timezone)

        return self.f[func](*args)

    def visit_Cast(self, op, *, arg, to):
        if to.is_interval():
            func = self.f[f"to_{_INTERVAL_SUFFIXES[to.unit.short]}"]
            return func(sg.cast(arg, to=self.type_mapper.from_ibis(dt.int32)))
        elif to.is_timestamp() and op.arg.dtype.is_integer():
            return self.f.to_timestamp(arg)

        return self.cast(arg, to)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_interval():
            if dtype.unit.short == "ns":
                raise com.UnsupportedOperationError(
                    f"{self.dialect} doesn't support nanosecond interval resolutions"
                )

            return sge.Interval(
                this=sge.convert(str(value)), unit=dtype.resolution.upper()
            )
        elif dtype.is_uuid():
            return self.cast(str(value), dtype)
        elif dtype.is_binary():
            return self.cast("".join(map("\\x{:02x}".format, value)), dtype)
        elif dtype.is_numeric():
            # cast non finite values to float because that's the behavior of
            # duckdb when a mixed decimal/float operation is performed
            #
            # float will be upcast to double if necessary by duckdb
            if not math.isfinite(value):
                return self.cast(
                    str(value), to=dt.float32 if dtype.is_decimal() else dtype
                )
            return self.cast(value, dtype)
        elif dtype.is_time():
            return self.f.make_time(
                value.hour, value.minute, value.second + value.microsecond / 1e6
            )
        elif dtype.is_timestamp():
            args = [
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second + value.microsecond / 1e6,
            ]

            funcname = "make_timestamp"

            if (tz := dtype.timezone) is not None:
                funcname += "tz"
                args.append(tz)

            return self.f[funcname](*args)
        elif dtype.is_struct():
            return sge.Struct.from_arg_list(
                [
                    sge.PropertyEQ(
                        this=sg.to_identifier(k, quoted=self.quoted),
                        expression=self.visit_Literal(
                            ops.Literal(v, field_dtype), value=v, dtype=field_dtype
                        ),
                    )
                    for field_dtype, (k, v) in zip(dtype.types, value.items())
                ]
            )
        else:
            return None

    def _neg_idx_to_pos(self, array, idx):
        arg_length = self.f.array_size(array)
        return self.if_(
            idx >= 0,
            idx,
            # Need to have the greatest here to handle the case where
            # abs(neg_index) > arg_length
            # e.g. where the magnitude of the negative index is greater than the
            # length of the array
            # You cannot index a[:-3] if a = [1, 2]
            arg_length + self.f.greatest(idx, -arg_length),
        )

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.dialect} only implements `pop` correlation coefficient"
            )

        # TODO: rewrite rule?
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    def visit_GeoConvert(self, op, *, arg, source, target):
        # 4th argument is to specify that the result is always_xy so that it
        # matches the behavior of the equivalent geopandas functionality
        return self.f.st_transform(arg, source, target, True)

    def visit_TimestampNow(self, op):
        """DuckDB current timestamp defaults to timestamp + tz."""
        return self.cast(super().visit_TimestampNow(op), dt.timestamp)

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        return self.f.regexp_extract(arg, pattern, index, dialect=self.dialect)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(
            arg, pattern, replacement, "g", dialect=self.dialect
        )

    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"
        return self.agg[funcname](arg, quantile, where=where)

    def visit_MultiQuantile(self, op, *, arg, quantile, where):
        return self.visit_Quantile(op, arg=arg, quantile=quantile, where=where)

    def visit_HexDigest(self, op, *, arg, how):
        if how in ("md5", "sha256"):
            return getattr(self.f, how)(arg)
        else:
            raise NotImplementedError(f"No available hashing function for {how}")

    def visit_StringConcat(self, op, *, arg):
        return reduce(lambda x, y: sge.DPipe(this=x, expression=y), arg)
