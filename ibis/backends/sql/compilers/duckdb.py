from __future__ import annotations

import math
from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import DuckDB

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import DuckDBType
from ibis.backends.sql.rewrites import exclude_nulls_from_array_collect
from ibis.util import gen_name

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


class DuckDBCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = DuckDB
    type_mapper = DuckDBType

    agg = AggGen(supports_filter=True)

    rewrites = (
        exclude_nulls_from_array_collect,
        *SQLGlotCompiler.rewrites,
    )

    LOWERED_OPS = {
        ops.Sample: None,
        ops.StringSlice: None,
    }

    SIMPLE_OPS = {
        ops.Arbitrary: "any_value",
        ops.ArrayPosition: "list_indexof",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.EndsWith: "suffix",
        ops.ExtractIsoYear: "isoyear",
        ops.IntegerRange: "range",
        ops.TimestampRange: "range",
        ops.MapLength: "cardinality",
        ops.Mode: "mode",
        ops.TimeFromHMS: "make_time",
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
                sge.PropertyEQ(this=k, expression=v[i])
                for k, v in zip(map(sge.convert, op.dtype.value_type.names), arg)
            ]
        )
        func = sge.Lambda(this=body, expressions=[i])
        zipped_arrays = self.f.list_apply(
            self.f.range(
                1,
                # DuckDB Range excludes upper bound
                self.f.greatest(*map(self.f.len, arg)) + 1,
            ),
            func,
        )
        # if any of the input arrays in arg are NULL, the result is NULL
        any_arg_null = sg.or_(*(arr.is_(NULL) for arr in arg))
        return self.if_(any_arg_null, NULL, zipped_arrays)

    def visit_Array(self, op, *, exprs):
        return self.cast(self.f.array(*exprs), op.dtype)

    def visit_Map(self, op, *, keys, values):
        # workaround for https://github.com/ibis-project/ibis/issues/8632
        return self.if_(
            sg.or_(keys.is_(NULL), values.is_(NULL)),
            NULL,
            self.f.map(
                self.cast(keys, op.keys.dtype), self.cast(values, op.values.dtype)
            ),
        )

    def visit_MapGet(self, op, *, arg, key, default):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.ifnull(
                self.f.list_extract(
                    self.if_(key.is_(NULL), NULL, self.f.element_at(arg, key)), 1
                ),
                default,
            ),
        )

    def visit_MapContains(self, op, *, arg, key):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.len(self.if_(key.is_(NULL), NULL, self.f.element_at(arg, key))).neq(
                0
            ),
        )

    def visit_MapKeys(self, op, *, arg):
        return self.if_(arg.is_(NULL), NULL, self.f.map_keys(arg))

    def visit_MapValues(self, op, *, arg):
        return self.if_(arg.is_(NULL), NULL, self.f.map_values(arg))

    def visit_MapMerge(self, op, *, left, right):
        return self.if_(
            sg.or_(left.is_(NULL), right.is_(NULL)),
            NULL,
            self.f.map_concat(left, right),
        )

    def visit_ToJSONMap(self, op, *, arg):
        return self.if_(
            self.f.json_type(arg).eq(sge.convert("OBJECT")),
            self.cast(self.cast(arg, dt.json), op.dtype),
            NULL,
        )

    def visit_ToJSONArray(self, op, *, arg):
        return self.if_(
            self.f.json_type(arg).eq(sge.convert("ARRAY")),
            self.cast(self.cast(arg, dt.json), op.dtype),
            NULL,
        )

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.if_(
            self.f.json_type(arg).eq(sge.convert("VARCHAR")),
            self.f.json_extract_string(arg, "$"),
            NULL,
        )

    def visit_UnwrapJSONInt64(self, op, *, arg):
        arg_type = self.f.json_type(arg)
        return self.if_(
            arg_type.isin(sge.convert("UBIGINT"), sge.convert("BIGINT")),
            self.cast(arg, op.dtype),
            NULL,
        )

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        arg_type = self.f.json_type(arg)
        return self.if_(
            arg_type.isin(
                sge.convert("UBIGINT"), sge.convert("BIGINT"), sge.convert("DOUBLE")
            ),
            self.cast(arg, op.dtype),
            NULL,
        )

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.if_(
            self.f.json_type(arg).eq(sge.convert("BOOLEAN")),
            self.cast(arg, op.dtype),
            NULL,
        )

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
        elif to.is_geospatial() and op.arg.dtype.is_binary():
            return self.f.st_geomfromwkb(arg)

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
            if dtype.is_floating() or dtype.is_integer():
                return sge.convert(value)
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
            return self.cast(
                sge.Struct.from_arg_list(
                    [
                        self.visit_Literal(
                            ops.Literal(v, field_dtype), value=v, dtype=field_dtype
                        )
                        for field_dtype, v in zip(dtype.types, value.values())
                    ]
                ),
                op.dtype,
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

    def visit_First(self, op, *, arg, where):
        cond = arg.is_(sg.not_(NULL, copy=False))
        where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.first(arg, where=where)

    def visit_Last(self, op, *, arg, where):
        cond = arg.is_(sg.not_(NULL, copy=False))
        where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.last(arg, where=where)

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

    def visit_Hash(self, op, *, arg):
        # duckdb's hash() returns a uint64, but ops.Hash is supposed to be int64
        # So do HASH(x)::BITSTRING::BIGINT
        raw = self.f.hash(arg)
        bitstring = sg.cast(sge.convert(raw), to=sge.DataType.Type.BIT, copy=False)
        int64 = sg.cast(bitstring, to=sge.DataType.Type.BIGINT, copy=False)
        return int64

    def visit_StringConcat(self, op, *, arg):
        return reduce(lambda x, y: sge.DPipe(this=x, expression=y), arg)

    def visit_StringSlice(self, op, *, arg, start, end):
        if start is not None:
            start += 1
        # workaround for https://github.com/duckdb/duckdb/issues/11431
        start = self.f.ifnull(start, 1)
        end = self.f.ifnull(end, -1)
        return self.f.array_slice(arg, start, end)

    def visit_StructField(self, op, *, arg, field):
        if not isinstance(op.arg, (ops.Field, sge.Struct)):
            # parenthesize anything that isn't a simple field access
            return sge.Dot(
                this=sge.paren(arg),
                expression=sg.to_identifier(field, quoted=self.quoted),
            )
        return super().visit_StructField(op, arg=arg, field=field)

    def visit_RandomScalar(self, op, **kwargs):
        return self.f.random()

    def visit_RandomUUID(self, op, **kwargs):
        return self.f.uuid()

    def visit_TypeOf(self, op, *, arg):
        return self.f.coalesce(self.f.nullif(self.f.typeof(arg), '"NULL"'), "NULL")

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        quoted = self.quoted
        # duckdb doesn't support specifying the table name of the column name
        # to drop, e.g., in SELECT t.* EXCLUDE (t.a) FROM t, the t.a bit
        #
        # technically it's not necessary, here's why
        #
        # if the table is specified then it's unambiguous when there are overlapping
        # column names, say, from a join, for example
        # (assuming t and s both have a column named `a`)
        #
        # SELECT t.* EXCLUDE (a), s.* FROM t JOIN s ON id
        #
        # This would exclude t.a and include s.a
        #
        # if it's a naked star projection from a join, like
        #
        # SELECT * EXCLUDE (a) FROM t JOIN s ON id
        #
        # then that means "exclude all columns named `a`"
        excludes = [sg.column(column, quoted=quoted) for column in columns_to_drop]
        star = sge.Star(**{"except": excludes})
        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)
        column = sge.Column(this=star, table=table)
        return sg.select(column).from_(parent)

    def visit_TableUnnest(
        self, op, *, parent, column, offset: str | None, keep_empty: bool
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(gen_name("table_unnest_column"), quoted=quoted)

        opname = op.column.name
        overlaps_with_parent = opname in op.parent.schema
        computed_column = column_alias.as_(opname, quoted=quoted)

        selcols = []

        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)

        if offset is not None:
            # TODO: clean this up once WITH ORDINALITY is supported in DuckDB
            # no need for struct_extract once that's upstream
            column = self.f.list_zip(column, self.f.range(self.f.len(column)))
            extract = self.f.struct_extract(column_alias, 1).as_(opname, quoted=quoted)

            if overlaps_with_parent:
                replace = sge.Column(this=sge.Star(replace=[extract]), table=table)
                selcols.append(replace)
            else:
                selcols.append(sge.Column(this=STAR, table=table))
                selcols.append(extract)

            selcols.append(
                self.f.struct_extract(column_alias, 2).as_(offset, quoted=quoted)
            )
        elif overlaps_with_parent:
            selcols.append(
                sge.Column(this=sge.Star(replace=[computed_column]), table=table)
            )
        else:
            selcols.append(sge.Column(this=STAR, table=table))
            selcols.append(computed_column)

        unnest = sge.Unnest(
            expressions=[column],
            alias=sge.TableAlias(
                this=sg.to_identifier(gen_name("table_unnest"), quoted=quoted),
                columns=[column_alias],
            ),
        )
        return (
            sg.select(*selcols)
            .from_(parent)
            .join(unnest, join_type="CROSS" if not keep_empty else "LEFT")
        )
