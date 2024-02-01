from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from public import public
from sqlglot.dialects import Oracle
from sqlglot.dialects.dialect import create_with_partitions_sql, rename_func

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import OracleType
from ibis.backends.base.sqlglot.rewrites import Window, replace_log2, replace_log10
from ibis.common.patterns import replace
from ibis.expr.analysis import p, x, y
from ibis.expr.rewrites import rewrite_sample


def _create_sql(self, expression: sge.Create) -> str:
    # TODO: should we use CREATE PRIVATE instead?  That will set an implicit
    # lower bound of Oracle 18c
    properties = expression.args.get("properties")
    temporary = any(
        isinstance(prop, sge.TemporaryProperty)
        for prop in (properties.expressions if properties else [])
    )

    kind = expression.args["kind"]
    if (obj := kind.upper()) in ("TABLE", "VIEW") and temporary:
        if expression.expression:
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} AS {self.sql(expression, 'expression')}"
        else:
            # TODO: why does autocommit not work here?  need to specify the ON COMMIT part...
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} ON COMMIT PRESERVE ROWS"

    return create_with_partitions_sql(self, expression)


Oracle.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Create: _create_sql,
    sge.Select: sg.transforms.preprocess([sg.transforms.eliminate_semi_and_anti_joins]),
}

# TODO: can delete this after bumping sqlglot version > 20.9.0
Oracle.Generator.TYPE_MAPPING |= {
    sge.DataType.Type.TIMETZ: "TIME",
    sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMP",
}
Oracle.Generator.TZ_TO_WITH_TIME_ZONE = True


@replace(p.WindowFunction(p.First(x, y)))
def rewrite_first(_, x, y):
    if y is not None:
        raise com.UnsupportedOperationError(
            "`first` aggregate over window does not support `where`"
        )
    return _.copy(func=ops.FirstValue(x))


@replace(p.WindowFunction(p.Last(x, y)))
def rewrite_last(_, x, y):
    if y is not None:
        raise com.UnsupportedOperationError(
            "`last` aggregate over window does not support `where`"
        )
    return _.copy(func=ops.LastValue(x))


@replace(p.WindowFunction(frame=x @ p.WindowFrame(order_by=())))
def rewrite_empty_order_by_window(_, x):
    return _.copy(frame=x.copy(order_by=(ibis.NA,)))


@replace(p.WindowFunction(p.RowNumber | p.NTile, x))
def exclude_unsupported_window_frame_from_row_number(_, x):
    return ops.Subtract(_.copy(frame=x.copy(start=None, end=None)), 1)


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All,
        x @ p.WindowFrame(start=None),
    )
)
def exclude_unsupported_window_frame_from_ops(_, x):
    return _.copy(frame=x.copy(start=None, end=None))


@public
class OracleCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "oracle"
    quoted = True
    type_mapper = OracleType
    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_ops,
        rewrite_first,
        rewrite_last,
        rewrite_empty_order_by_window,
        rewrite_sample,
        replace_log2,
        replace_log10,
        *SQLGlotCompiler.rewrites,
    )

    NAN = sge.Literal.number("binary_double_nan")
    """Backend's NaN literal."""

    POS_INF = sge.Literal.number("binary_double_infinity")
    """Backend's positive infinity literal."""

    NEG_INF = sge.Literal.number("-binary_double_infinity")
    """Backend's negative infinity literal."""

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg) for arg in args)
        return func(*args)

    @staticmethod
    def _generate_groups(groups):
        return groups

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Equals)
    def visit_Equals(self, op, *, left, right):
        # Oracle didn't have proper boolean types until recently and we handle them
        # as integers so we end up with things like "t0"."bool_col" = 1 (for True)
        # but then if we are testing that a boolean column IS True, it gets rendered as
        # "t0"."bool_col" = 1 = 1
        # so intercept that and change it to WHERE (bool_col = 1)
        # TODO(gil): there must be a better way to do this
        if op.dtype.is_boolean() and isinstance(right, sge.Boolean):
            if right.this:
                return left
            else:
                return sg.not_(left)
        return super().visit_Equals(op, left=left, right=right)

    @visit_node.register(ops.IsNull)
    def visit_IsNull(self, op, *, arg):
        # TODO(gil): find a better way to handle this
        # but CASE WHEN (bool_col = 1) IS NULL isn't valid and we can simply check if
        # bool_col is null
        if isinstance(arg, sge.EQ):
            return arg.this.is_(NULL)
        return arg.is_(NULL)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        # avoid casting NULL -- oracle handling for these casts is... complicated
        if value is None:
            return NULL
        elif dtype.is_timestamp() or dtype.is_time():
            if getattr(dtype, "timezone", None) is not None:
                return self.f.to_timestamp_tz(
                    value.isoformat(), 'YYYY-MM-DD"T"HH24:MI:SS.FF6TZH:TZM'
                )
            else:
                return self.f.to_timestamp(
                    value.isoformat(), 'YYYY-MM-DD"T"HH24:MI:SS.FF6'
                )
        elif dtype.is_date():
            return self.f.to_date(
                f"{value.year:04d}-{value.month:02d}-{value.day:02d}", "FXYYYY-MM-DD"
            )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        elif dtype.is_interval():
            if dtype.unit.short in ("Y", "M"):
                return self.f.numtoyminterval(value, dtype.unit.name)
            elif dtype.unit.short in ("D", "h", "m", "s"):
                return self.f.numtodsinterval(value, dtype.unit.name)
            else:
                raise com.UnsupportedOperationError(
                    f"Intervals with precision {dtype.unit.name} not supported in Oracle."
                )

        return super().visit_Literal(op, value=value, dtype=dtype)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        if to.is_interval():
            # CASTing to an INTERVAL in Oracle requires specifying digits of
            # precision that are a pain.  There are two helper functions that
            # should be used instead.
            if to.unit.short in ("D", "h", "m", "s"):
                return self.f.numtodsinterval(arg, to.unit.name)
            elif to.unit.short in ("Y", "M"):
                return self.f.numtoyminterval(arg, to.unit.name)
            else:
                raise com.UnsupportedArgumentError(
                    f"Interval {to.unit.name} not supported by Oracle"
                )
        return self.cast(arg, to)

    @visit_node.register(ops.Limit)
    def visit_Limit(self, op, *, parent, n, offset):
        # push limit/offset into subqueries
        if isinstance(parent, sge.Subquery) and parent.this.args.get("limit") is None:
            result = parent.this
            alias = parent.alias
        else:
            result = sg.select(STAR).from_(parent)
            alias = None

        if isinstance(n, int):
            result = result.limit(n)
        elif n is not None:
            raise com.UnsupportedArgumentError(
                "No support for dynamic limit in the Oracle backend."
            )
            # TODO: re-enable this for dynamic limits
            # but it should be paired with offsets working
            # result = result.where(C.ROWNUM <= sg.select(n).from_(parent).subquery())
        else:
            assert n is None, n
            if self.no_limit_value is not None:
                result = result.limit(self.no_limit_value)

        assert offset is not None, "offset is None"

        if offset > 0:
            raise com.UnsupportedArgumentError(
                "No support for limit offsets in the Oracle backend."
            )

        if alias is not None:
            return result.subquery(alias)
        return result

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return sg.cast(arg, to="date")

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.NAN)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        # Not using FuncGen here because of dotted function call
        return sg.func("dbms_random.value")

    @visit_node.register(ops.Pi)
    def visit_Pi(self, op):
        return self.f.acos(-1)

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1 / self.f.tan(arg)

    @visit_node.register(ops.Degrees)
    def visit_Degrees(self, op, *, arg):
        return 180 * arg / self.visit_node(ops.Pi())

    @visit_node.register(ops.Radians)
    def visit_Radians(self, op, *, arg):
        return self.visit_node(ops.Pi()) * arg / 180

    @visit_node.register(ops.Modulus)
    def visit_Modulus(self, op, *, left, right):
        return self.f.mod(left, right)

    @visit_node.register(ops.Levenshtein)
    def visit_Levenshtein(self, op, *, left, right):
        # Not using FuncGen here because of dotted function call
        return sg.func("utl_match.edit_distance", left, right)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return self.f.substr(arg, 0, self.f.length(start)).eq(start)

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return self.f.substr(arg, -1 * self.f.length(end), self.f.length(end)).eq(end)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise NotImplementedError("`end` is not implemented")

        sub_string = substr

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, sub_string)
            # TODO(gil): why, oh why, does this need an extra +1 on the end?
            return sg.case().when(pos > 0, pos - 1 + start).else_(-1) + 1

        return self.f.instr(arg, sub_string)

    @visit_node.register(ops.StrRight)
    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        return self.if_(
            index.eq(0),
            self.f.regexp_substr(arg, pattern),
            self.f.regexp_substr(arg, pattern, 1, 1, "cn", index),
        )

    @visit_node.register(ops.RegexReplace)
    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return sge.RegexpReplace(this=arg, expression=pattern, replacement=replacement)

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) > 0

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.concat(*toolz.interpose(sep, arg))

    ## Aggregate stuff

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, where, how):
        if how == "sample":
            raise ValueError(
                "Oracle only implements population correlation coefficient"
            )
        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, where, how):
        if how == "sample":
            return self.agg.covar_samp(left, right, where=where)
        return self.agg.covar_pop(left, right, where=where)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.visit_Quantile(op, arg=arg, quantile=0.5, where=where)

    @visit_node.register(ops.Quantile)
    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"

        if where is not None:
            arg = self.if_(where, arg)

        expr = sge.WithinGroup(
            this=self.f[funcname](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        return expr

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg)

        return sge.Count(this=sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.count(self.if_(where, 1, NULL))
        return self.f.count(STAR)

    @visit_node.register(ops.IdenticalTo)
    def visit_IdenticalTo(self, op, *, left, right):
        # sqlglot NullSafeEQ uses "is not distinct from" which isn't supported in oracle
        return (
            sg.case()
            .when(left.eq(right).or_(left.is_(NULL).and_(right.is_(NULL))), 0)
            .else_(1)
            .eq(0)
        )

    @visit_node.register(ops.Xor)
    def visit_Xor(self, op, *, left, right):
        return (left.or_(right)).and_(sg.not_(left.and_(right)))

    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.DateTruncate)
    def visit_DateTruncate(self, op, *, arg, unit):
        trunc_unit_mapping = {
            "Y": "year",
            "M": "MONTH",
            "W": "IW",
            "D": "DDD",
            "h": "HH",
            "m": "MI",
        }

        timestamp_unit_mapping = {
            "s": "SS",
            "ms": "SS.FF3",
            "us": "SS.FF6",
            "ns": "SS.FF9",
        }

        if (unyt := timestamp_unit_mapping.get(unit.short)) is not None:
            # Oracle only has trunc(DATE) and that can't do sub-minute precision, but we can
            # handle those separately.
            return self.f.to_timestamp(
                self.f.to_char(arg, f"YYYY-MM-DD HH24:MI:{unyt}"),
                f"YYYY-MM-DD HH24:MI:{unyt}",
            )

        if (unyt := trunc_unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

        return self.f.trunc(arg, unyt)

    @visit_node.register(Window)
    def visit_Window(self, op, *, how, func, start, end, group_by, order_by):
        # Oracle has two (more?) types of analytic functions you can use inside OVER.
        #
        # The first group accepts an "analytic clause" which is decomposed into the
        # PARTITION BY, ORDER BY and the windowing clause (e.g. ROWS BETWEEN
        # UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING).  These are the "full" window functions.
        #
        # The second group accepts an _optional_ PARTITION BY clause and a _required_ ORDER BY clause.
        # If you try to pass, for instance, LEAD(col, 1) OVER() AS "val", this will error.
        #
        # The list of functions which accept the full analytic clause (and so
        # accept a windowing clause) are those functions which are marked with
        # an asterisk at the bottom of this page (yes, Oracle thinks this is
        # a reasonable way to demarcate them):
        # https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Analytic-Functions.html
        #
        # (Side note: these unordered window function queries were not erroring
        # in the SQLAlchemy Oracle backend but they were raising AssertionErrors.
        # This is because the SQLAlchemy Oracle dialect automatically inserts an
        # ORDER BY whether you ask it to or not.)
        #
        # If the windowing clause is omitted, the default is
        # RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        #
        # I (@gforsyth) believe that this is the windowing range applied to the
        # analytic functions (like LEAD, LAG, CUME_DIST) which don't allow
        # specifying a windowing clause.
        #
        # This allowance for specifying a windowing clause is handled below by
        # explicitly listing the ops which correspond to the analytic functions
        # that accept it.

        if type(op.func) in (
            # TODO: figure out REGR_* functions and also manage this list better
            # Allowed windowing clause functions
            ops.Mean,  # "avg",
            ops.Correlation,  # "corr",
            ops.Count,  # "count",
            ops.Covariance,  # "covar_pop", "covar_samp",
            ops.FirstValue,  # "first_value",
            ops.LastValue,  # "last_value",
            ops.Max,  # "max",
            ops.Min,  # "min",
            ops.NthValue,  # "nth_value",
            ops.StandardDev,  # "stddev","stddev_pop","stddev_samp",
            ops.Sum,  # "sum",
            ops.Variance,  # "var_pop","var_samp","variance",
        ):
            if start is None:
                start = {}
            if end is None:
                end = {}

            start_value = start.get("value", "UNBOUNDED")
            start_side = start.get("side", "PRECEDING")
            end_value = end.get("value", "UNBOUNDED")
            end_side = end.get("side", "FOLLOWING")

            spec = sge.WindowSpec(
                kind=how.upper(),
                start=start_value,
                start_side=start_side,
                end=end_value,
                end_side=end_side,
                over="OVER",
            )
        elif not order_by:
            # For other analytic functions, ORDER BY is required
            raise com.UnsupportedOperationError(
                f"Function {op.func.name} cannot be used in Oracle without an order_by."
            )
        else:
            # and no windowing clause is supported, so set the spec to None.
            spec = None

        order = sge.Order(expressions=order_by) if order_by else None

        spec = self._minimize_spec(op.start, op.end, spec)

        return sge.Window(this=func, partition_by=group_by, order=order, spec=spec)

    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.Array)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.First)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Mode)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.TimeTruncate)
    @visit_node.register(ops.Bucket)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.TimestampNow)
    @visit_node.register(ops.TimestampFromYMDHMS)
    @visit_node.register(ops.TimeFromHMS)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.DayOfWeekIndex)
    @visit_node.register(ops.DayOfWeekName)
    @visit_node.register(ops.DateDiff)
    @visit_node.register(ops.ExtractEpochSeconds)
    @visit_node.register(ops.ExtractWeekOfYear)
    @visit_node.register(ops.ExtractDayOfYear)
    @visit_node.register(ops.RowID)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.BitAnd: "bit_and_agg",
    ops.BitOr: "bit_or_agg",
    ops.BitXor: "bit_xor_agg",
    ops.BitwiseAnd: "bitand",
    ops.Hash: "hash",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.StringAscii: "ascii",
    ops.Strip: "trim",
    ops.Hash: "ora_hash",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @OracleCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @OracleCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(OracleCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
