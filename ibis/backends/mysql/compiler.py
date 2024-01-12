from __future__ import annotations

import string
from functools import partial, reduce, singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.dialects import MySQL
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import MySQLType
from ibis.backends.base.sqlglot.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_empty_order_by_window,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)
from ibis.common.patterns import replace
from ibis.expr.rewrites import p, rewrite_sample, y

MySQL.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.StddevPop: rename_func("stddev_pop"),
    sge.StddevSamp: rename_func("stddev_samp"),
    sge.RegexpLike: (
        lambda _, e: f"({e.this.sql('mysql')} RLIKE {e.expression.sql('mysql')})"
    ),
}


@replace(p.Limit)
def rewrite_limit(_, **kwargs):
    """Rewrite limit for MySQL to include a large upper bound.

    From the MySQL docs @ https://dev.mysql.com/doc/refman/8.0/en/select.html

    > To retrieve all rows from a certain offset up to the end of the result
    > set, you can use some large number for the second parameter. This statement
    > retrieves all rows from the 96th row to the last:
    >
    > SELECT * FROM tbl LIMIT 95,18446744073709551615;
    """
    if _.n is None and _.offset is not None:
        some_large_number = (1 << 64) - 1
        return _.copy(n=some_large_number)
    return _


@replace(p.WindowFunction(p.MinRank | p.DenseRank, y @ p.WindowFrame(start=None)))
def exclude_unsupported_window_frame_from_rank(_, y):
    return ops.Subtract(
        _.copy(frame=y.copy(start=None, end=0, order_by=y.order_by or (ops.NULL,))), 1
    )


@public
class MySQLCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "mysql"
    type_mapper = MySQLType
    rewrites = (
        rewrite_limit,
        rewrite_sample,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )
    quoted = True

    @property
    def NAN(self):
        raise NotImplementedError("MySQL does not support NaN")

    @property
    def POS_INF(self):
        raise NotImplementedError("MySQL does not support Infinity")

    NEG_INF = POS_INF

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

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

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if (from_.is_json() or from_.is_string()) and to.is_json():
            # MariaDB does not support casting to JSON because it's an alias
            # for TEXT (except when casting of course!)
            return arg
        elif from_.is_integer() and to.is_interval():
            return self.visit_IntervalFromInteger(
                ops.IntervalFromInteger(op.arg, unit=to.unit), arg=arg, unit=to.unit
            )
        elif from_.is_integer() and to.is_timestamp():
            return self.f.from_unixtime(arg)
        return super().visit_Cast(op, arg=arg, to=to)

    @visit_node.register(ops.TimestampDiff)
    def visit_TimestampDiff(self, op, *, left, right):
        return self.f.timestampdiff(
            sge.Var(this="SECOND"), right, left, dialect=self.dialect
        )

    @visit_node.register(ops.DateDiff)
    def visit_DateDiff(self, op, *, left, right):
        return self.f.timestampdiff(
            sge.Var(this="DAY"), right, left, dialect=self.dialect
        )

    @visit_node.register(ops.ApproxCountDistinct)
    def visit_ApproxCountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg)
        return self.f.count(sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.cast(where, op.dtype))
        return self.f.count(STAR)

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg)
        return self.f.count(sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, arg, where):
        if where is not None:
            raise com.UnsupportedOperationError(
                "Filtered table count distinct is not supported in MySQL"
            )
        func = partial(sg.column, table=arg.alias_or_name, quoted=self.quoted)
        return self.f.count(
            sge.Distinct(expressions=list(map(func, op.arg.schema.keys())))
        )

    @visit_node.register(ops.GroupConcat)
    def visit_GroupConcat(self, op, *, arg, sep, where):
        if not isinstance(op.sep, ops.Literal):
            raise com.UnsupportedOperationError(
                "Only string literal separators are supported"
            )
        if where is not None:
            arg = self.if_(where, arg)
        return self.f.group_concat(arg, sep)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.dayofweek(arg) + 5) % 7

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        # avoid casting NULL: the set of types allowed by MySQL and
        # MariaDB when casting is a strict subset of allowed types in other
        # contexts like CREATE TABLE
        if value is None:
            return NULL
        return super().visit_Literal(op, value=value, dtype=dtype)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_decimal() and not value.is_finite():
            raise com.UnsupportedOperationError(
                "MySQL does not support NaN or infinity"
            )
        elif dtype.is_binary():
            return self.f.unhex(value.hex())
        elif dtype.is_date():
            return self.f.date(value.isoformat())
        elif dtype.is_timestamp():
            return self.f.timestamp(value.isoformat())
        elif dtype.is_time():
            return self.f.maketime(
                value.hour, value.minute, value.second + value.microsecond / 1e6
            )
        elif dtype.is_array() or dtype.is_struct() or dtype.is_map():
            raise com.UnsupportedBackendType(
                "MySQL does not support arrays, structs or maps"
            )
        elif dtype.is_string():
            return sge.convert(value.replace("\\", "\\\\"))
        return None

    @visit_node.register(ops.JSONGetItem)
    def visit_JSONGetItem(self, op, *, arg, index):
        if op.index.dtype.is_integer():
            path = self.f.concat("$[", self.cast(index, dt.string), "]")
        else:
            path = self.f.concat("$.", index)
        return self.f.json_extract(arg, path)

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.f.str_to_date(
            self.f.concat(
                self.f.lpad(year, 4, "0"),
                self.f.lpad(month, 2, "0"),
                self.f.lpad(day, 2, "0"),
            ),
            "%Y%m%d",
        )

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values):
        return self.f.find_in_set(needle, self.f.concat_ws(",", values))

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        to = sge.DataType(this=sge.DataType.Type.BINARY)
        return self.f.right(arg, self.f.char_length(end)).eq(sge.Cast(this=end, to=to))

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        to = sge.DataType(this=sge.DataType.Type.BINARY)
        return self.f.left(arg, self.f.length(start)).eq(sge.Cast(this=start, to=to))

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern):
        return arg.rlike(pattern)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        extracted = self.f.regexp_substr(arg, pattern)
        return self.if_(
            arg.rlike(pattern),
            self.if_(
                index.eq(0),
                extracted,
                self.f.regexp_replace(
                    extracted, pattern, rf"\\{index.sql(self.dialect)}"
                ),
            ),
            NULL,
        )

    @visit_node.register(ops.Equals)
    def visit_Equals(self, op, *, left, right):
        if op.left.dtype.is_string():
            assert op.right.dtype.is_string(), op.right.dtype
            to = sge.DataType(this=sge.DataType.Type.BINARY)
            return sge.Cast(this=left, to=to).eq(right)
        return super().visit_Equals(op, left=left, right=right)

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) > 0

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise NotImplementedError(
                "`end` argument is not implemented for MySQL `StringValue.find`"
            )
        substr = sge.Cast(this=substr, to=sge.DataType(this=sge.DataType.Type.BINARY))

        if start is not None:
            return self.f.locate(substr, arg, start + 1)
        return self.f.locate(substr, arg)

    @visit_node.register(ops.Capitalize)
    def visit_Capitalize(self, op, *, arg):
        return self.f.concat(
            self.f.upper(self.f.left(arg, 1)), self.f.lower(self.f.substr(arg, 2))
        )

    def visit_LRStrip(self, op, *, arg, position):
        return reduce(
            lambda arg, char: self.f.trim(this=arg, position=position, expression=char),
            map(
                partial(self.cast, to=dt.string),
                map(self.f.unhex, map(self.f.hex, string.whitespace.encode())),
            ),
            arg,
        )

    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimestampTruncate)
    def visit_DateTimestampTruncate(self, op, *, arg, unit):
        truncate_formats = {
            "s": "%Y-%m-%d %H:%i:%s",
            "m": "%Y-%m-%d %H:%i:00",
            "h": "%Y-%m-%d %H:00:00",
            "D": "%Y-%m-%d",
            # 'W': 'week',
            "M": "%Y-%m-01",
            "Y": "%Y-01-01",
        }
        if (format := truncate_formats.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {op.unit}")
        return self.f.date_format(arg, format)

    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.DateDelta)
    def visit_DateTimeDelta(self, op, *, left, right, part):
        return self.f.timestampdiff(
            sge.Var(this=part.this), right, left, dialect=self.dialect
        )

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg):
        return self.f.floor(self.f.extract(sge.Var(this="microsecond"), arg) / 1_000)

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.floor(self.f.extract(sge.Var(this="microsecond"), arg))

    @visit_node.register(ops.Strip)
    def visit_Strip(self, op, *, arg):
        return self.visit_LRStrip(op, arg=arg, position="BOTH")

    @visit_node.register(ops.LStrip)
    def visit_LStrip(self, op, *, arg):
        return self.visit_LRStrip(op, arg=arg, position="LEADING")

    @visit_node.register(ops.RStrip)
    def visit_RStrip(self, op, *, arg):
        return self.visit_LRStrip(op, arg=arg, position="TRAILING")

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return sge.Interval(this=arg, unit=sge.convert(op.resolution.upper()))

    @visit_node.register(ops.TimestampAdd)
    def visit_TimestampAdd(self, op, *, left, right):
        if op.right.dtype.unit.short == "ms":
            right = sge.Interval(
                this=right.this * 1_000, unit=sge.Var(this="MICROSECOND")
            )
        return self.f.date_add(left, right, dialect=self.dialect)

    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.Array)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.First)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.Mode)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.RegexReplace)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.StructColumn)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.Translate)
    @visit_node.register(ops.Unnest)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.DayOfWeekName: "dayname",
    ops.Log10: "log10",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.StringContains: "instr",
    ops.ExtractWeekOfYear: "weekofyear",
    ops.ExtractEpochSeconds: "unix_timestamp",
    ops.ExtractDayOfYear: "dayofyear",
    ops.Strftime: "date_format",
    ops.StringToTimestamp: "str_to_date",
    ops.Log2: "log2",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @MySQLCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @MySQLCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(MySQLCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
