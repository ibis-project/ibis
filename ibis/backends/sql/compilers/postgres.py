from __future__ import annotations

import inspect
import json
import string
import textwrap
from functools import partial, reduce
from itertools import takewhile
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import PostgresType
from ibis.backends.sql.dialects import Postgres
from ibis.backends.sql.rewrites import (
    lower_sample,
    split_select_distinct_with_order_by,
    subtract_one_from_array_map_filter_index,
)
from ibis.common.exceptions import InvalidDecoratorError
from ibis.util import gen_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    import ibis.expr.types as ir


def _verify_source_line(func_name: str, line: str):
    if line.startswith("@"):
        raise InvalidDecoratorError(func_name, line)
    return line


class PostgresUDFNode(ops.Value):
    shape = rlz.shape_like("args")


class PostgresCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Postgres
    type_mapper = PostgresType
    rewrites = (subtract_one_from_array_map_filter_index, *SQLGlotCompiler.rewrites)
    post_rewrites = (split_select_distinct_with_order_by,)

    agg = AggGen(supports_filter=True, supports_order_by=True)

    NAN = sge.Literal.number("'NaN'::double precision")
    POS_INF = sge.Literal.number("'Inf'::double precision")
    NEG_INF = sge.Literal.number("'-Inf'::double precision")

    LOWERED_OPS = {ops.Sample: lower_sample(physical_tables_only=True)}

    UNSUPPORTED_OPS = (
        ops.RowID,
        ops.TimeDelta,
        ops.ArrayFlatten,
        ops.Kurtosis,
    )

    SIMPLE_OPS = {
        ops.Arbitrary: "first",  # could use any_value for postgres>=16
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
        ops.MapContains: "jsonb_contains",
        ops.RegexSearch: "regexp_like",
        ops.TimeFromHMS: "make_time",
        ops.RandomUUID: "gen_random_uuid",
    }

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
            if typ.is_map() or typ.is_json()
        )

        if conversions:
            table_expr = table_expr.mutate(**conversions)
        return super().to_sqlglot(table_expr, limit=limit, params=params)

    def _compile_python_udf(self, udf_node: ops.ScalarUDF):
        config = udf_node.__config__
        func = udf_node.__func__
        func_name = func.__name__

        lines, _ = inspect.getsourcelines(func)
        iter_lines = iter(lines)

        function_premable_lines = list(
            takewhile(lambda line: not line.lstrip().startswith("def "), iter_lines)
        )

        if len(function_premable_lines) > 1:
            raise InvalidDecoratorError(
                name=func_name, lines="".join(function_premable_lines)
            )

        source = textwrap.dedent(
            "".join(map(partial(_verify_source_line, func_name), iter_lines))
        ).strip()

        type_mapper = self.type_mapper
        argnames = udf_node.argnames
        args = ", ".join(argnames)
        name = type(udf_node).__name__
        argsig = ", ".join(argnames)
        raw_args = [
            f"json.loads({argname})" if arg.dtype.is_map() else argname
            for argname, arg in zip(argnames, udf_node.args)
        ]
        args = ", ".join(raw_args)
        call = f"{name}({args})"
        defn = """\
CREATE OR REPLACE FUNCTION {ident}({signature})
RETURNS {return_type}
LANGUAGE {language}
AS $$
{json_import}
def {name}({argsig}):
{source}
return {call}
$$""".format(
            ident=self.__sql_name__(udf_node),
            signature=", ".join(
                f"{argname} {type_mapper.to_string(arg.dtype)}"
                for argname, arg in zip(argnames, udf_node.args)
            ),
            return_type=type_mapper.to_string(udf_node.dtype),
            language=config.get("language", "plpython3u"),
            json_import=(
                "import json"
                if udf_node.dtype.is_map()
                or any(arg.dtype.is_map() for arg in udf_node.args)
                else ""
            ),
            name=name,
            argsig=argsig,
            source=textwrap.indent(source, " " * 4),
            call=call if not udf_node.dtype.is_map() else f"json.dumps({call})",
        )
        return defn

    def visit_Mode(self, op, *, arg, where):
        expr = self.f.mode()
        expr = sge.WithinGroup(
            this=expr,
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    def _argminmax(self, op, *, arg, key, where, desc: bool):
        cond = key.is_(sg.not_(NULL))
        where = cond if where is None else sge.And(this=cond, expression=where)

        agg = self.agg.array_agg(
            sge.Ordered(this=sge.Order(this=arg, expressions=[key]), desc=desc),
            where=where,
        )
        return sge.paren(agg, copy=False)[0]

    def visit_ArgMin(self, op, *, arg, key, where):
        return self._argminmax(op, arg=arg, key=key, where=where, desc=False)

    def visit_ArgMax(self, op, *, arg, key, where):
        return self._argminmax(op, arg=arg, key=key, where=where, desc=True)

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

    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"
        expr = sge.WithinGroup(
            this=self.f[funcname](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    visit_MultiQuantile = visit_Quantile
    visit_ApproxQuantile = visit_Quantile
    visit_ApproxMultiQuantile = visit_Quantile

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
        return reduce(
            lambda x, y: self.if_(
                x.is_(NULL).or_(y.is_(NULL)), NULL, self.f.array_cat(x, y)
            ),
            map(partial(self.cast, to=op.dtype), arg),
        )

    def visit_ArrayContains(self, op, *, arg, other):
        arg_dtype = op.arg.dtype
        # ArrayContainsAll introduced in 24, keep backcompat if it doesn't exist
        cls = getattr(sge, "ArrayContainsAll", sge.ArrayContains)
        return cls(
            this=self.cast(arg, arg_dtype),
            expression=self.f.array(self.cast(other, arg_dtype.value_type)),
        )

    def visit_ArrayFilter(self, op, *, arg, body, param, index):
        if index is None:
            alias = param
        else:
            alias = sge.TableAlias(this=sg.to_identifier("_"), columns=[param])

        return self.f.array(
            sg.select(sg.column(param, quoted=self.quoted))
            .from_(sge.Unnest(expressions=[arg], alias=alias, offset=index))
            .where(body)
        )

    def visit_ArrayMap(self, op, *, arg, body, param, index):
        if index is None:
            alias = param
        else:
            alias = sge.TableAlias(this=sg.to_identifier("_"), columns=[param])
        return self.f.array(
            sg.select(body).from_(
                sge.Unnest(expressions=[arg], alias=alias, offset=index)
            )
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

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        if distinct:
            arg = sge.Distinct(expressions=[arg])
        return self.agg.array_agg(arg, where=where, order_by=order_by)

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the postgres backend"
            )
        return self.agg.first(arg, where=where, order_by=order_by)

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the postgres backend"
            )
        return self.agg.last(arg, where=where, order_by=order_by)

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
        k, v = map(sg.to_identifier, "kv")
        regular = (
            sg.select(self.f.jsonb_object_agg(k, v))
            .from_(
                sg.select(
                    self.f.unnest(keys).as_(k), self.f.unnest(values).as_(v)
                ).subquery()
            )
            .subquery()
        )
        return self.if_(keys.is_(NULL).or_(values.is_(NULL)), NULL, regular)

    def visit_MapLength(self, op, *, arg):
        return (
            sg.select(self.f.count(sge.Star()))
            .from_(self.f.jsonb_object_keys(arg))
            .subquery()
        )

    def visit_MapGet(self, op, *, arg, key, default):
        if op.dtype.is_null():
            return NULL
        else:
            return self.cast(
                self.if_(
                    self.f.jsonb_contains(arg, key),
                    self.f.jsonb_extract_path_text(arg, key),
                    default,
                ),
                op.dtype,
            )

    def visit_MapMerge(self, op, *, left, right):
        return sge.DPipe(this=left, expression=right)

    def visit_MapKeys(self, op, *, arg):
        return self.if_(
            arg.is_(NULL), NULL, self.f.array(sg.select(self.f.jsonb_object_keys(arg)))
        )

    def visit_MapValues(self, op, *, arg):
        col = gen_name("json_each_col")
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.array(
                sg.select(
                    sge.Dot(
                        this=sg.to_identifier(col),
                        expression=sg.to_identifier("value", quoted=True),
                    )
                ).from_(self.f.jsonb_each(arg).as_(col))
            ),
        )

    def visit_TypeOf(self, op, *, arg):
        typ = self.cast(self.f.pg_typeof(arg), dt.string)
        return self.if_(
            typ.eq(sge.convert("unknown")),
            "null" if op.arg.dtype.is_null() else "text",
            typ,
        )

    def visit_Round(self, op, *, arg, digits):
        dtype = op.dtype

        if dtype.is_integer():
            result = self.f.round(arg)
        else:
            result = self.f.round(self.cast(arg, dt.decimal), digits)

        return self.cast(result, dtype)

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
        elif dtype.is_map():
            return sge.Cast(
                this=sge.convert(json.dumps(value)),
                to=sge.DataType(this=sge.DataType.Type.JSONB),
            )
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

    def _make_interval(self, arg, unit):
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
        elif from_.is_numeric() and to.is_timestamp():
            arg = self.f.to_timestamp(arg)
            if (timezone := to.timezone) is not None:
                arg = self.f.timezone(timezone, arg)
            return arg
        elif from_.is_string() and to.is_binary():
            # Postgres and Python use the words "decode" and "encode" in
            # opposite ways, sweet!
            return self.f.decode(arg, "escape")
        elif from_.is_binary() and to.is_string():
            return self.f.encode(arg, "escape")

        return super().visit_Cast(op, arg=arg, to=to)

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

    def visit_JoinLink(self, op, *, how, table, predicates):
        if how == "asof":
            # Convert asof join to a lateral left join

            # The asof match condition is always the first predicate
            match_condition, *predicates = predicates
            on = sg.and_(*predicates) if predicates else None

            return sge.Join(
                this=sge.Lateral(
                    this=sge.Subquery(
                        this=sg.select(sge.Star())
                        .from_(table)
                        .where(match_condition)
                        # the ordering for the subquery depends on whether we
                        # want to pick the one row with the largest or smallest
                        # value that fulfills the match condition
                        .order_by(
                            match_condition.expression.asc()
                            if match_condition.key in {"lte", "lt"}
                            else match_condition.expression.desc()
                        )
                        .limit(1)
                    )
                ).as_(table.alias_or_name),
                kind="left",
                on=on,
            )

        return super().visit_JoinLink(op, how=how, table=table, predicates=predicates)

    def visit_TableUnnest(
        self,
        op,
        *,
        parent,
        column,
        column_name: str,
        offset: str | None,
        keep_empty: bool,
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(gen_name("table_unnest_column"), quoted=quoted)

        parent_alias = parent.alias_or_name

        parent_schema = op.parent.schema
        overlaps_with_parent = column_name in parent_schema
        computed_column = column_alias.as_(column_name, quoted=quoted)

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

    def _unnest(self, expression, *, as_):
        alias = sge.TableAlias(columns=[sg.to_identifier(as_)])
        return sge.Unnest(expressions=[expression], alias=alias)

    def _array_reduction(self, *, arg, reduction):
        name = sg.to_identifier(gen_name(f"pg_arr_{reduction}"))
        return (
            sg.select(self.f[reduction](name))
            .from_(self._unnest(arg, as_=name))
            .subquery()
        )

    def visit_ArrayStringJoin(self, op, *, arg, sep):
        dtype = op.arg.dtype
        return self.f.array_to_string(
            self.f.nullif(self.cast(arg, dtype), self.cast(self.f.array(), dtype)),
            sep,
        )

    def visit_ArrayMin(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="min")

    def visit_ArrayMax(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="max")

    def visit_ArraySum(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="sum")

    def visit_ArrayMean(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="avg")

    def visit_ArrayAny(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="bool_or")

    def visit_ArrayAll(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="bool_and")

    def visit_ArrayMode(self, op, *, arg):
        name = sg.to_identifier(gen_name("pg_arr_mode"))
        expr = sge.WithinGroup(
            this=self.f.mode(),
            expression=sge.Order(expressions=[sge.Ordered(this=name)]),
        )
        return sg.select(expr).from_(self._unnest(arg, as_=name)).subquery()

    def visit_StringToTime(self, op, *, arg, format_str):
        return self.cast(self.f.str_to_time(arg, format_str), to=dt.time)


compiler = PostgresCompiler()
