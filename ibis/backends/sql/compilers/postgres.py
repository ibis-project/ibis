from __future__ import annotations

import string
from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import PostgresType
from ibis.backends.sql.dialects import Postgres
from ibis.backends.sql.rewrites import exclude_nulls_from_array_collect
from ibis.util import gen_name


class PostgresUDFNode(ops.Value):
    shape = rlz.shape_like("args")


class PostgresCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Postgres
    type_mapper = PostgresType

    rewrites = (exclude_nulls_from_array_collect, *SQLGlotCompiler.rewrites)

    agg = AggGen(supports_filter=True)

    NAN = sge.Literal.number("'NaN'::double precision")
    POS_INF = sge.Literal.number("'Inf'::double precision")
    NEG_INF = sge.Literal.number("'-Inf'::double precision")

    UNSUPPORTED_OPS = (
        ops.RowID,
        ops.TimeDelta,
        ops.ArrayFlatten,
    )

    SIMPLE_OPS = {
        ops.Arbitrary: "first",  # could use any_value for postgres>=16
        ops.ArrayCollect: "array_agg",
        ops.ArrayRemove: "array_remove",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.GeoArea: "st_area",
        ops.GeoAsBinary: "st_asbinary",
        ops.GeoAsEWKB: "st_asewkb",
        ops.GeoAsEWKT: "st_asewkt",
        ops.GeoAsText: "st_astext",
        ops.GeoAzimuth: "st_azimuth",
        ops.GeoBuffer: "st_buffer",
        ops.GeoCentroid: "st_centroid",
        ops.GeoContains: "st_contains",
        ops.GeoContainsProperly: "st_contains",
        ops.GeoCoveredBy: "st_coveredby",
        ops.GeoCovers: "st_covers",
        ops.GeoCrosses: "st_crosses",
        ops.GeoDFullyWithin: "st_dfullywithin",
        ops.GeoDWithin: "st_dwithin",
        ops.GeoDifference: "st_difference",
        ops.GeoDisjoint: "st_disjoint",
        ops.GeoDistance: "st_distance",
        ops.GeoEndPoint: "st_endpoint",
        ops.GeoEnvelope: "st_envelope",
        ops.GeoEquals: "st_equals",
        ops.GeoGeometryN: "st_geometryn",
        ops.GeoGeometryType: "st_geometrytype",
        ops.GeoIntersection: "st_intersection",
        ops.GeoIntersects: "st_intersects",
        ops.GeoIsValid: "st_isvalid",
        ops.GeoLength: "st_length",
        ops.GeoLineLocatePoint: "st_linelocatepoint",
        ops.GeoLineMerge: "st_linemerge",
        ops.GeoLineSubstring: "st_linesubstring",
        ops.GeoNPoints: "st_npoints",
        ops.GeoOrderingEquals: "st_orderingequals",
        ops.GeoOverlaps: "st_overlaps",
        ops.GeoPerimeter: "st_perimeter",
        ops.GeoSRID: "st_srid",
        ops.GeoSetSRID: "st_setsrid",
        ops.GeoSimplify: "st_simplify",
        ops.GeoStartPoint: "st_startpoint",
        ops.GeoTouches: "st_touches",
        ops.GeoTransform: "st_transform",
        ops.GeoUnaryUnion: "st_union",
        ops.GeoUnion: "st_union",
        ops.GeoWithin: "st_within",
        ops.GeoX: "st_x",
        ops.GeoY: "st_y",
        ops.MapContains: "exist",
        ops.MapKeys: "akeys",
        ops.MapValues: "avals",
        ops.RegexSearch: "regexp_like",
        ops.TimeFromHMS: "make_time",
    }

    def visit_RandomUUID(self, op, **kwargs):
        return self.f.gen_random_uuid()

    def visit_Mode(self, op, *, arg, where):
        expr = self.f.mode()
        expr = sge.WithinGroup(
            this=expr,
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    def visit_ArgMinMax(self, op, *, arg, key, where, desc: bool):
        conditions = [arg.is_(sg.not_(NULL)), key.is_(sg.not_(NULL))]

        if where is not None:
            conditions.append(where)

        agg = self.agg.array_agg(
            sge.Ordered(this=sge.Order(this=arg, expressions=[key]), desc=desc),
            where=sg.and_(*conditions),
        )
        return sge.paren(agg, copy=False)[0]

    def visit_ArgMin(self, op, *, arg, key, where):
        return self.visit_ArgMinMax(op, arg=arg, key=key, where=where, desc=False)

    def visit_ArgMax(self, op, *, arg, key, where):
        return self.visit_ArgMinMax(op, arg=arg, key=key, where=where, desc=True)

    def visit_Sum(self, op, *, arg, where):
        arg = (
            self.cast(self.cast(arg, dt.int32), op.dtype)
            if op.arg.dtype.is_boolean()
            else arg
        )
        return self.agg.sum(arg, where=where)

    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.cast(sge.convert("NaN"), op.arg.dtype))

    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    def visit_CountDistinctStar(self, op, *, where, arg):
        # use a tuple because postgres doesn't accept COUNT(DISTINCT a, b, c, ...)
        #
        # this turns the expression into COUNT(DISTINCT ROW(a, b, c, ...))
        row = sge.Tuple(
            expressions=list(
                map(partial(sg.column, quoted=self.quoted), op.arg.schema.keys())
            )
        )
        return self.agg.count(sge.Distinct(expressions=[row]), where=where)

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

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.visit_Median(op, arg=arg, where=where)

    def visit_Median(self, op, *, arg, where):
        return self.visit_Quantile(op, arg=arg, quantile=sge.convert(0.5), where=where)

    def visit_ApproxCountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def visit_Range(self, op, *, start, stop, step):
        def zero_value(dtype):
            if dtype.is_interval():
                return self.f.make_interval()
            return 0

        def interval_sign(v):
            zero = self.f.make_interval()
            return sge.Case(
                ifs=[
                    self.if_(v.eq(zero), 0),
                    self.if_(v < zero, -1),
                    self.if_(v > zero, 1),
                ],
                default=NULL,
            )

        def _sign(value, dtype):
            if dtype.is_interval():
                return interval_sign(value)
            return self.f.sign(value)

        step_dtype = op.step.dtype
        return self.if_(
            sg.and_(
                self.f.nullif(step, zero_value(step_dtype)).is_(sg.not_(NULL)),
                _sign(step, step_dtype).eq(_sign(stop - start, step_dtype)),
            ),
            self.f.array_remove(
                self.f.array(
                    sg.select(STAR).from_(self.f.generate_series(start, stop, step))
                ),
                stop,
            ),
            self.cast(self.f.array(), op.dtype),
        )

    visit_IntegerRange = visit_TimestampRange = visit_Range

    def visit_StringConcat(self, op, *, arg):
        return reduce(lambda x, y: sge.DPipe(this=x, expression=y), arg)

    def visit_ArrayConcat(self, op, *, arg):
        return reduce(self.f.array_cat, map(partial(self.cast, to=op.dtype), arg))

    def visit_ArrayContains(self, op, *, arg, other):
        arg_dtype = op.arg.dtype
        # ArrayContainsAll introduced in 24, keep backcompat if it doesn't exist
        cls = getattr(sge, "ArrayContainsAll", sge.ArrayContains)
        return cls(
            this=self.cast(arg, arg_dtype),
            expression=self.f.array(self.cast(other, arg_dtype.value_type)),
        )

    def visit_ArrayFilter(self, op, *, arg, body, param):
        return self.f.array(
            sg.select(sg.column(param, quoted=self.quoted))
            .from_(sge.Unnest(expressions=[arg], alias=param))
            .where(body)
        )

    def visit_ArrayMap(self, op, *, arg, body, param):
        return self.f.array(
            sg.select(body).from_(sge.Unnest(expressions=[arg], alias=param))
        )

    def visit_ArrayPosition(self, op, *, arg, other):
        t = sge.Unnest(expressions=[arg], alias="value", offset=True)
        idx = sg.column("ordinality")
        value = sg.column("value")
        return self.f.coalesce(
            sg.select(idx).from_(t).where(value.eq(other)).limit(1).subquery(), 0
        )

    def visit_ArraySort(self, op, *, arg):
        return self.f.array(
            sg.select("x").from_(sge.Unnest(expressions=[arg], alias="x")).order_by("x")
        )

    def visit_ArrayRepeat(self, op, *, arg, times):
        i = sg.to_identifier("i")
        length = self.f.cardinality(arg)
        return self.f.array(
            sg.select(arg[i % length + 1]).from_(
                self.f.generate_series(0, length * times - 1).as_(i.name)
            )
        )

    def visit_ArrayDistinct(self, op, *, arg):
        return self.if_(
            arg.is_(NULL), NULL, self.f.array(sg.select(self.f.explode(arg)).distinct())
        )

    def visit_ArrayUnion(self, op, *, left, right):
        return self.f.anon.array(
            sg.union(sg.select(self.f.explode(left)), sg.select(self.f.explode(right)))
        )

    def visit_ArrayIntersect(self, op, *, left, right):
        return self.f.anon.array(
            sg.intersect(
                sg.select(self.f.explode(left)), sg.select(self.f.explode(right))
            )
        )

    def visit_Log2(self, op, *, arg):
        return self.cast(
            self.f.log(
                self.cast(sge.convert(2), dt.decimal),
                arg if op.arg.dtype.is_decimal() else self.cast(arg, dt.decimal),
            ),
            op.dtype,
        )

    def visit_Log(self, op, *, arg, base):
        if base is not None:
            if not op.base.dtype.is_decimal():
                base = self.cast(base, dt.decimal)
        else:
            base = self.cast(sge.convert(self.f.exp(1)), dt.decimal)

        if not op.arg.dtype.is_decimal():
            arg = self.cast(arg, dt.decimal)
        return self.cast(self.f.log(base, arg), op.dtype)

    def visit_StructField(self, op, *, arg, field):
        idx = op.arg.dtype.names.index(field) + 1
        # postgres doesn't have anonymous structs :(
        #
        # this works around ibis not having a way to tell sqlglot to transform
        # an exploded array(row) into the equivalent unnest(t) _ (col1, ..., colN)
        # element
        #
        # but also postgres should really support anonymous structs
        return self.cast(
            self.f.jsonb_extract_path(self.f.to_jsonb(arg), f"f{idx:d}"), op.dtype
        )

    def json_typeof(self, op, arg):
        b = "b" * op.arg.dtype.binary
        return self.f[f"json{b}_typeof"](arg)

    def json_extract_path_text(self, op, arg, *rest):
        b = "b" * op.arg.dtype.binary
        return self.f[f"json{b}_extract_path_text"](
            arg,
            *rest,
            # this is apparently how you pass in no additional arguments to
            # a variadic function, see the "Variadic Function Resolution"
            # section in
            # https://www.postgresql.org/docs/current/typeconv-func.html
            sge.Var(this="VARIADIC ARRAY[]::TEXT[]"),
        )

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.if_(
            self.json_typeof(op, arg).eq(sge.convert("string")),
            self.json_extract_path_text(op, arg),
            NULL,
        )

    def visit_UnwrapJSONInt64(self, op, *, arg):
        text = self.json_extract_path_text(op, arg)
        return self.if_(
            self.json_typeof(op, arg).eq(sge.convert("number")),
            self.cast(
                self.if_(self.f.regexp_like(text, r"^\d+$", "g"), text, NULL), op.dtype
            ),
            NULL,
        )

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        text = self.json_extract_path_text(op, arg)
        return self.if_(
            self.json_typeof(op, arg).eq(sge.convert("number")),
            self.cast(text, op.dtype),
            NULL,
        )

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.if_(
            self.json_typeof(op, arg).eq(sge.convert("boolean")),
            self.cast(self.json_extract_path_text(op, arg), op.dtype),
            NULL,
        )

    def visit_JSONGetItem(self, op, *, arg, index):
        if op.arg.dtype.binary:
            return self.f.jsonb_extract_path(arg, self.cast(index, dt.string))
        return super().visit_JSONGetItem(op, arg=arg, index=index)

    def visit_StructColumn(self, op, *, names, values):
        return self.f.row(*map(self.cast, values, op.dtype.types))

    def visit_ToJSONArray(self, op, *, arg):
        b = "b" * op.arg.dtype.binary
        return self.if_(
            self.json_typeof(op, arg).eq(sge.convert("array")),
            self.f.array(sg.select(STAR).from_(self.f[f"json{b}_array_elements"](arg))),
            NULL,
        )

    def visit_Map(self, op, *, keys, values):
        # map(["a", "b"], NULL) results in {"a": NULL, "b": NULL} in regular postgres,
        # so we need to modify it to return NULL instead
        regular = self.f.map(keys, values)
        return self.if_(values.is_(NULL), NULL, regular)

    def visit_MapLength(self, op, *, arg):
        return self.f.cardinality(self.f.akeys(arg))

    def visit_MapGet(self, op, *, arg, key, default):
        return self.if_(
            self.f.exist(arg, key),
            self.f.jsonb_extract_path_text(self.f.to_jsonb(arg), key),
            default,
        )

    def visit_MapMerge(self, op, *, left, right):
        return sge.DPipe(this=left, expression=right)

    def visit_TypeOf(self, op, *, arg):
        typ = self.cast(self.f.pg_typeof(arg), dt.string)
        return self.if_(
            typ.eq(sge.convert("unknown")),
            "null" if op.arg.dtype.is_null() else "text",
            typ,
        )

    def visit_Round(self, op, *, arg, digits):
        if digits is None:
            return self.f.round(arg)

        result = self.f.round(self.cast(arg, dt.decimal), digits)
        if op.arg.dtype.is_decimal():
            return result
        return self.cast(result, dt.float64)

    def visit_Modulus(self, op, *, left, right):
        # postgres doesn't allow modulus of double precision values, so upcast and
        # then downcast later if necessary
        if not op.dtype.is_integer():
            left = self.cast(left, dt.decimal)
            right = self.cast(right, dt.decimal)

        result = left % right
        if op.dtype.is_float64():
            return self.cast(result, dt.float64)
        else:
            return result

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        pattern = self.f.concat("(", pattern, ")")
        matches = self.f.regexp_match(arg, pattern)
        return self.if_(arg.rlike(pattern), sge.paren(matches, copy=False)[index], NULL)

    def visit_FindInSet(self, op, *, needle, values):
        return self.f.coalesce(
            self.f.array_position(self.f.array(*values), needle),
            0,
        )

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
            return self.cast("".join(map(r"\x{:0>2x}".format, value)), dt.binary)
        elif dtype.is_time():
            to_int32 = partial(self.cast, to=dt.int32)
            to_float64 = partial(self.cast, to=dt.float64)

            return self.f.make_time(
                to_int32(value.hour),
                to_int32(value.minute),
                to_float64(value.second + value.microsecond / 1e6),
            )
        elif dtype.is_json():
            return self.cast(value, dt.json)
        return None

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        to_int32 = partial(self.cast, to=dt.int32)
        return self.f.make_timestamp(
            to_int32(year),
            to_int32(month),
            to_int32(day),
            to_int32(hours),
            to_int32(minutes),
            self.cast(seconds, dt.float64),
        )

    def visit_DateFromYMD(self, op, *, year, month, day):
        to_int32 = partial(self.cast, to=dt.int32)
        return self.f.datefromparts(to_int32(year), to_int32(month), to_int32(day))

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        origin = self.f.make_timestamp(
            *map(partial(self.cast, to=dt.int32), (1970, 1, 1, 0, 0, 0))
        )

        if offset is not None:
            origin += offset

        return self.f.date_bin(interval, arg, origin)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.cast(self.f.extract("dow", arg) + 6, dt.int16) % 7

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.trim(self.f.to_char(arg, "Day"), string.whitespace)

    def visit_ExtractSecond(self, op, *, arg):
        return self.cast(self.f.floor(self.f.extract("second", arg)), op.dtype)

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.cast(
            self.f.floor(self.f.extract("millisecond", arg)) % 1_000, op.dtype
        )

    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.extract("microsecond", arg) % 1_000_000

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract("doy", arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract("week", arg)

    def visit_ExtractIsoYear(self, op, *, arg):
        return self.f.extract("isoyear", arg)

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.extract("epoch", arg)

    def visit_ArrayIndex(self, op, *, arg, index):
        index = self.if_(index < 0, self.f.cardinality(arg) + index, index)
        return sge.paren(arg, copy=False)[index]

    def visit_ArraySlice(self, op, *, arg, start, stop):
        neg_to_pos_index = lambda n, index: self.if_(index < 0, n + index, index)

        arg_length = self.f.cardinality(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, neg_to_pos_index(arg_length, start))

        if stop is None:
            stop = arg_length
        else:
            stop = neg_to_pos_index(arg_length, stop)

        slice_expr = sge.Slice(this=start + 1, expression=stop)
        return sge.paren(arg, copy=False)[slice_expr]

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        plural = unit.plural
        if plural == "minutes":
            plural = "mins"
            arg = self.cast(arg, dt.int32)
        elif plural == "seconds":
            plural = "secs"
            arg = self.cast(arg, dt.float64)
        elif plural == "milliseconds":
            plural = "secs"
            arg /= 1e3
        elif plural == "microseconds":
            plural = "secs"
            arg /= 1e6
        elif plural == "nanoseconds":
            plural = "secs"
            arg /= 1e9
        else:
            arg = self.cast(arg, dt.int32)

        key = sg.to_identifier(plural)

        return self.f.make_interval(sge.Kwarg(this=key, expression=arg))

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if from_.is_timestamp() and to.is_integer():
            return self.f.extract("epoch", arg)
        elif from_.is_integer() and to.is_timestamp():
            arg = self.f.to_timestamp(arg)
            if (timezone := to.timezone) is not None:
                arg = self.f.timezone(timezone, arg)
            return arg
        elif from_.is_integer() and to.is_interval():
            unit = to.unit
            return self.visit_IntervalFromInteger(
                ops.IntervalFromInteger(op.arg, unit), arg=arg, unit=unit
            )
        elif from_.is_string() and to.is_binary():
            # Postgres and Python use the words "decode" and "encode" in
            # opposite ways, sweet!
            return self.f.decode(arg, "escape")
        elif from_.is_binary() and to.is_string():
            return self.f.encode(arg, "escape")

        return self.cast(arg, op.to)

    visit_TryCast = visit_Cast

    def visit_Hash(self, op, *, arg):
        arg_dtype = op.arg.dtype

        if arg_dtype.is_int16():
            return self.f.hashint2extended(arg, 0)
        elif arg_dtype.is_int32():
            return self.f.hashint4extended(arg, 0)
        elif arg_dtype.is_int64():
            return self.f.hashint8extended(arg, 0)
        elif arg_dtype.is_float32():
            return self.f.hashfloat4extended(arg, 0)
        elif arg_dtype.is_float64():
            return self.f.hashfloat8extended(arg, 0)
        elif arg_dtype.is_string():
            return self.f.hashtextextended(arg, 0)
        elif arg_dtype.is_macaddr():
            return self.f.hashmacaddr8extended(arg, 0)

        raise com.UnsupportedOperationError(
            f"Hash({arg_dtype!r}) operation is not supported in the "
            f"{self.dialect} backend"
        )

    def visit_TableUnnest(
        self, op, *, parent, column, offset: str | None, keep_empty: bool
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(gen_name("table_unnest_column"), quoted=quoted)

        parent_alias = parent.alias_or_name

        opname = op.column.name
        parent_schema = op.parent.schema
        overlaps_with_parent = opname in parent_schema
        computed_column = column_alias.as_(opname, quoted=quoted)

        selcols = []

        if overlaps_with_parent:
            column_alias_or_name = column.alias_or_name
            selcols.extend(
                sg.column(col, table=parent_alias, quoted=quoted)
                if col != column_alias_or_name
                else computed_column
                for col in parent_schema.names
            )
        else:
            selcols.append(
                sge.Column(
                    this=STAR, table=sg.to_identifier(parent_alias, quoted=quoted)
                )
            )
            selcols.append(computed_column)

        if offset is not None:
            offset_name = offset
            offset = sg.to_identifier(offset_name, quoted=quoted)
            selcols.append((offset - 1).as_(offset_name, quoted=quoted))

        unnest = sge.Unnest(
            expressions=[column],
            alias=sge.TableAlias(
                this=sg.to_identifier(gen_name("table_unnest"), quoted=quoted),
                columns=[column_alias],
            ),
            offset=offset,
        )

        return (
            sg.select(*selcols)
            .from_(parent)
            .join(
                unnest,
                on=None if not keep_empty else sge.convert(True),
                join_type="CROSS" if not keep_empty else "LEFT",
            )
        )
