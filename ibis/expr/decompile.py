from __future__ import annotations

import builtins
import collections
import enum
import functools
import io
import itertools
import keyword

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.graph import Graph
from ibis.expr.rewrites import simplify
from ibis.util import experimental

_method_overrides = {
    ops.ApproxCountDistinct: "approx_nunique",
    ops.ArrayCollect: "collect",
    ops.CountDistinct: "nunique",
    ops.CountStar: "count",
    ops.DateTruncate: "truncate",
    ops.EndsWith: "endswith",
    ops.ExtractDay: "day",
    ops.ExtractDayOfYear: "day_of_year",
    ops.ExtractEpochSeconds: "epoch_seconds",
    ops.ExtractHour: "hour",
    ops.ExtractMicrosecond: "microsecond",
    ops.ExtractMillisecond: "millisecond",
    ops.ExtractMinute: "minute",
    ops.ExtractMonth: "month",
    ops.ExtractQuarter: "quarter",
    ops.ExtractSecond: "second",
    ops.ExtractWeekOfYear: "week_of_year",
    ops.ExtractYear: "year",
    ops.Intersection: "intersect",
    ops.IntervalFromInteger: "as_interval",
    ops.IsNull: "isnull",
    ops.LStrip: "lstrip",
    ops.Lowercase: "lower",
    ops.MinRank: "rank",
    ops.NthValue: "nth",
    ops.NullIf: "nullif",
    ops.RStrip: "rstrip",
    ops.RegexSearch: "re_search",
    ops.StandardDev: "std",
    ops.StartsWith: "startswith",
    ops.StringContains: "contains",
    ops.StringFind: "find",
    ops.StringLength: "length",
    ops.StringSplit: "split",
    ops.StringSQLILike: "ilike",
    ops.StringSQLLike: "like",
    ops.Substring: "substr",
    ops.TimeTruncate: "truncate",
    ops.TimestampTruncate: "truncate",
    ops.Uppercase: "upper",
    ops.Variance: "var",
}


def _to_snake_case(camel_case):
    """Convert a camelCase string to snake_case."""
    result = list(camel_case[:1].lower())
    for char in camel_case[1:]:
        if char.isupper():
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _get_method_name(op):
    typ = op.__class__
    try:
        return _method_overrides[typ]
    except KeyError:
        return _to_snake_case(typ.__name__)


def _maybe_add_parens(op, string):
    if isinstance(op, ops.Binary):
        return f"({string})"
    elif isinstance(string, CallStatement):
        return string.args
    else:
        return string


class CallStatement:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return f"{self.func}({self.args})"


@functools.singledispatch
def translate(op, *args, **kwargs):
    """Translate an ibis operation into a Python expression."""
    raise NotImplementedError(op)


@translate.register(ops.TimestampNow)
def now(_):
    return "ibis.now()"


@translate.register(ops.DateNow)
def today(_):
    return "ibis.today()"


def _render_raw(value):
    """Render a non-node argument (a Python value stored on the op) as code."""
    if isinstance(value, enum.Enum):
        value = value.value
    return repr(value)


def _nondefault_kwargs(op, kwargs):
    """Keep the keyword arguments whose value differs from the op's default."""
    params = op.__signature__.parameters
    result = []
    for name, rendered in kwargs.items():
        raw = getattr(op, name)
        if raw is None or raw == () or raw == params[name].default:
            continue
        if isinstance(raw, (tuple, list)):
            rendered = f"[{_inline(rendered)}]"
        elif not isinstance(raw, ops.Node):
            rendered = _render_raw(raw)
        result.append((name, rendered))
    return result


@translate.register(ops.Value)
def value(op, *args, **kwargs):
    method = _get_method_name(op)
    kwargs = _nondefault_kwargs(op, kwargs)

    if args:
        this, *args = args
    else:
        (_, this), *kwargs = kwargs

    # if there is a single keyword argument prefer to pass that as positional
    if not args and len(kwargs) == 1:
        args = [kwargs[0][1]]
        kwargs = []

    args = ", ".join(map(str, args))
    kwargs = ", ".join(f"{k}={v}" for k, v in kwargs)
    parameters = ", ".join(filter(None, [args, kwargs]))

    return f"{this}.{method}({parameters})"


@translate.register(ops.Coalesce)
@translate.register(ops.Greatest)
@translate.register(ops.Least)
def variadic(op, arg):
    return f"ibis.{_get_method_name(op)}({_inline(arg)})"


@translate.register(ops.StringConcat)
def string_concat(op, arg):
    first, *rest = arg
    return f"{first}.concat({_inline(rest)})"


@translate.register(ops.Not)
def not_(op, arg):
    # parenthesized like a binary operator: ``~x.name('y')`` would apply the
    # method first
    return f"(~{_maybe_add_parens(op.arg, arg)})"


@translate.register(ops.IdenticalTo)
def identical_to(op, left, right):
    return f"{left}.identical_to({right})"


@translate.register(ops.ArrayIndex)
def array_index(op, arg, index):
    return f"{arg}[{index}]"


@translate.register(ops.DayOfWeekIndex)
def day_of_week_index(op, arg):
    return f"{arg}.day_of_week.index()"


@translate.register(ops.DateDelta)
@translate.register(ops.TimeDelta)
@translate.register(ops.TimestampDelta)
def temporal_delta(op, left, right, part):
    unit = repr(op.part.value) if isinstance(op.part, ops.Literal) else part
    return f"{left}.delta({right}, unit={unit})"


@translate.register(ops.Negate)
def negate(op, arg):
    return f"(-{_maybe_add_parens(op.arg, arg)})"


@translate.register(ops.ExistsSubquery)
def exists_subquery(op, rel, rendered, outer):
    # ``(pred).any()`` is the documented way to spell a correlated EXISTS. It
    # takes the outer table from the relation it is evaluated against, so it
    # rebuilds only when the predicates span exactly the inner relation and
    # that outer one, and hold no subquery of their own
    if isinstance(op.rel, ops.Filter) and outer is not None:
        preds = op.rel.predicates
        relations = set().union(*(pred.relations for pred in preds))
        if (
            len(relations) == 2
            and relations == {op.rel.parent, outer}
            and not any(pred.find(ops.Subquery, filter=ops.Value) for pred in preds)
        ):
            pieces = [_maybe_add_parens(pred, rendered[pred]) for pred in preds]
            return f"({' & '.join(pieces)}).any()"
    return f"({rel}.count() > 0)"


def _single_field(rel):
    """Return (parent, name) if `rel` selects a single column of its parent."""
    if isinstance(rel, ops.Project) and len(rel.values) == 1:
        (field,) = rel.values.values()
        if isinstance(field, ops.Field) and field.rel == rel.parent:
            return rel.parent, field.name
    return None


@translate.register(ops.ScalarSubquery)
def scalar_subquery(op, rel, rendered):
    if (single := _single_field(op.rel)) is not None:
        parent, name = single
        return f"{rendered[parent]}.{name}.as_scalar()"
    return f"{rel}.as_scalar()"


@translate.register(ops.InSubquery)
def in_subquery(op, rel, needle, rendered):
    if (single := _single_field(op.rel)) is not None:
        parent, name = single
        return f"{needle}.isin({rendered[parent]}.{name})"
    (name,) = op.rel.schema.names
    return f"{needle}.isin({rel}.{name})"


@translate.register(ops.RankBase)
def rank(op):
    return f"ibis.{_get_method_name(op)}()"


@translate.register(ops.NTile)
def ntile(op, buckets):
    return f"ibis.ntile({buckets})"


@translate.register(ops.WindowBoundary)
def window_boundary(op, value, preceding):
    if isinstance(value, CallStatement):
        value = value.args
    if preceding and getattr(op.value, "value", None) != 0:
        return f"-{value}"
    return value


@translate.register(ops.WindowFunction)
def window_function(op, func, how, start, end, group_by, order_by):
    params = []
    if group_by:
        params.append(f"group_by=[{_inline(group_by)}]")
    if order_by:
        params.append(f"order_by=[{_inline(order_by)}]")
    if start is not None or end is not None:
        params.append(f"{how}=({start}, {end})")
    return f"{func}.over({', '.join(params)})"


@translate.register(ops.ScalarParameter)
def scalar_parameter(op, dtype, counter):
    return f"ibis.param({str(dtype)!r})"


@translate.register(ops.UnboundTable)
@translate.register(ops.DatabaseTable)
def table(op, schema, name, **kwargs):
    fields = dict(zip(schema.names, map(str, schema.types)))
    return f"ibis.table(name={name!r}, schema={fields})"


def _try_unwrap(stmt):
    if len(stmt) == 1:
        return stmt[0]
    else:
        stmt = map(str, stmt)
        values = ", ".join(stmt)
        return f"[{values}]"


def _wrap_alias(values, rendered):
    result = []
    for k, v in values.items():
        text = rendered[k]
        if v.name != k:
            if isinstance(v, ops.Binary):
                text = f"({text}).name({k!r})"
            else:
                text = f"{text}.name({k!r})"
        result.append(text)
    return result


def _inline(args):
    return ", ".join(map(str, args))


@translate.register(ops.Project)
def project(op, parent, values):
    out = f"{parent}"
    if not values:
        return out

    values = _wrap_alias(op.values, values)
    return f"{out}.select({_inline(values)})"


@translate.register(ops.Filter)
def filter_(op, parent, predicates):
    out = f"{parent}"
    if predicates:
        out = f"{out}.filter({_inline(predicates)})"
    return out


@translate.register(ops.Sort)
def sort(op, parent, keys):
    out = f"{parent}"
    if keys:
        out = f"{out}.order_by({_inline(keys)})"
    return out


@translate.register(ops.Aggregate)
def aggregation(op, parent, groups, metrics):
    groups = _wrap_alias(op.groups, groups)
    metrics = _wrap_alias(op.metrics, metrics)
    if groups and metrics:
        return f"{parent}.aggregate([{_inline(metrics)}], by=[{_inline(groups)}])"
    elif metrics:
        return f"{parent}.aggregate([{_inline(metrics)}])"
    elif groups:
        return f"{parent}.aggregate(by=[{_inline(groups)}])"
    else:
        raise ValueError("No metrics to aggregate")


@translate.register(ops.Distinct)
def distinct(op, parent):
    return f"{parent}.distinct()"


@translate.register(ops.DropColumns)
def drop(op, parent, columns_to_drop):
    return f"{parent}.drop({_inline(map(repr, columns_to_drop))})"


@translate.register(ops.SelfReference)
def self_reference(op, parent, identifier):
    return f"{parent}.view()"


@translate.register(ops.JoinReference)
def join_reference(op, parent, identifier):
    return parent


@translate.register(ops.JoinLink)
def join_link(op, table, predicates, how):
    if not predicates:
        return f".{how}_join({table})"
    return f".{how}_join({table}, {_try_unwrap(predicates)})"


@translate.register(ops.JoinChain)
def join(op, first, rest, values):
    calls = "".join(rest)
    pieces = [f"{first}{calls}"]
    if values:
        values = _wrap_alias(op.values, values)
        pieces.append(f"select({_inline(values)})")
    result = ".".join(pieces)
    return result


# the default of ``distinct`` differs between the set operation methods
_set_operation_defaults = {
    ops.Union: False,
    ops.Intersection: True,
    ops.Difference: True,
}


@translate.register(ops.Set)
def union(op, left, right, distinct):
    method = _get_method_name(op)
    if op.distinct != _set_operation_defaults[type(op)]:
        return f"{left}.{method}({right}, distinct={op.distinct})"
    return f"{left}.{method}({right})"


@translate.register(ops.Limit)
def limit(op, parent, n, offset):
    if offset:
        return f"{parent}.limit({n}, offset={offset})"
    else:
        return f"{parent}.limit({n})"


@translate.register(ops.Field)
def table_column(op, rel, name):
    if name.isidentifier():
        return f"{rel}.{name}"
    return f"{rel}[{name!r}]"


@translate.register(ops.SortKey)
def sort_key(op, arg, ascending, nulls_first):
    method = "asc" if ascending else "desc"
    call = f"{arg}.{method}"
    if nulls_first:
        return f"{call}(nulls_first={nulls_first})"
    return f"{call}()"


@translate.register(ops.Reduction)
def reduction(op, arg, where, **kwargs):
    method = _get_method_name(op)
    kwargs = _nondefault_kwargs(op, {**kwargs, "where": where})
    params = ", ".join(f"{k}={v}" for k, v in kwargs)
    return f"{arg}.{method}({params})"


@translate.register(ops.Alias)
def alias(op, arg, name):
    arg = _maybe_add_parens(op.arg, arg)
    return f"{arg}.name({name!r})"


@translate.register(ops.Constant)
def constant(op, **kwargs):
    method = _get_method_name(op)
    return f"ibis.{method}()"


@translate.register(ops.Literal)
def literal(op, value, dtype):
    inferred = ibis.literal(value)

    if isinstance(op.dtype, dt.Timestamp):
        return f'ibis.timestamp("{value}")'
    elif isinstance(op.dtype, dt.Date):
        return f"ibis.date({value!r})"
    elif isinstance(op.dtype, dt.Interval):
        return f"ibis.interval({value!r}, {op.dtype.unit.value!r})"
    elif isinstance(op.dtype, dt.Decimal):
        return f"ibis.literal({str(value)!r}, {str(dtype)!r})"
    elif inferred.type() != op.dtype:
        return f"ibis.literal({value!r}, {str(dtype)!r})"
    else:
        # prefer plain python literal values if the inferred datatype is the
        # same; a ``CallStatement`` is what the operand and argument rules
        # unwrap to the plain value
        return CallStatement("ibis.literal", repr(value))


@translate.register(ops.Cast)
def cast(op, arg, to):
    return f"{arg}.cast({str(to)!r})"


@translate.register(ops.TryCast)
def try_cast(op, arg, to):
    return f"{arg}.try_cast({str(to)!r})"


@translate.register(ops.Between)
def between(op, arg, lower_bound, upper_bound):
    return f"{arg}.between({lower_bound}, {upper_bound})"


@translate.register(ops.IfElse)
def ifelse(op, bool_expr, true_expr, false_null_expr):
    return f"{bool_expr}.ifelse({true_expr}, {false_null_expr})"


@translate.register(ops.SimpleCase)
@translate.register(ops.SearchedCase)
def switch_cases(op, cases, results, default, base=None):
    namespace = f"{base}" if base else "ibis"
    case_strs = [f"({case}, {result})" for case, result in zip(cases, results)]
    cases_str = ", ".join(case_strs)
    else_str = f", else_={default}" if default is not None else ""
    return f"{namespace}.cases({cases_str}{else_str})"


_infix_ops = {
    ops.Equals: "==",
    ops.NotEquals: "!=",
    ops.GreaterEqual: ">=",
    ops.Greater: ">",
    ops.LessEqual: "<=",
    ops.Less: "<",
    ops.And: "&",
    ops.Or: "|",
    ops.Xor: "^",
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.FloorDivide: "//",
    ops.Power: "**",
    ops.Modulus: "%",
    ops.TimestampAdd: "+",
    ops.TimestampSub: "-",
    ops.TimestampDiff: "-",
    ops.DateAdd: "+",
    ops.DateSub: "-",
    ops.DateDiff: "-",
    ops.TimeAdd: "+",
    ops.TimeSub: "-",
    ops.TimeDiff: "-",
    ops.IntervalAdd: "+",
    ops.IntervalSubtract: "-",
    ops.IntervalMultiply: "*",
}


@translate.register(ops.Binary)
def binary(op, left, right):
    operator = _infix_ops[type(op)]
    if isinstance(left, CallStatement):
        # a plain Python value on the left would fold with a plain right
        # operand, or swap places with an ibis one through the reflected
        # operator; keep it an ibis expression
        left = str(left)
    left = _maybe_add_parens(op.left, left)
    right = _maybe_add_parens(op.right, right)
    return _maybe_add_parens(op, f"{left} {operator} {right}")


@translate.register(ops.InValues)
def isin(op, value, options):
    return f"{value}.isin([{_inline(options)}])"


@translate.register(ops.DummyTable)
def dummy_table(op, values):
    values = _wrap_alias(op.values, values)
    first, *rest = values
    out = f"{first}.as_table()"
    if rest:
        out = f"{out}.mutate({_inline(rest)})"
    return out


class CodeContext:
    always_assign = (
        ops.ScalarParameter,
        ops.Aggregate,
        ops.PhysicalTable,
        ops.SelfReference,
    )

    always_ignore = (
        ops.JoinLink,
        ops.JoinReference,
        ops.Field,
        dt.Primitive,
        dt.Variadic,
        dt.Temporal,
    )
    shorthands = {
        ops.Aggregate: "agg",
        ops.Literal: "lit",
        ops.ScalarParameter: "param",
        ops.Project: "p",
        ops.Relation: "r",
        ops.Filter: "f",
        ops.Sort: "s",
    }

    # rules that look at how another node was rendered
    needs_rendered = (ops.ExistsSubquery, ops.ScalarSubquery, ops.InSubquery)

    def __init__(self, assign_result_to="result"):
        self.assign_result_to = assign_result_to
        self._shorthand_counters = collections.defaultdict(itertools.count)
        self.rendered = {}
        # the relation each EXISTS is evaluated against, by subquery node
        self.exists_outer = {}

    def variable_for(self, node):
        klass = type(node)
        if isinstance(node, ops.Relation) and hasattr(node, "name"):
            name = node.name
        elif klass in self.shorthands:
            name = self.shorthands[klass]
        else:
            name = klass.__name__.lower()

        if keyword.iskeyword(name) or hasattr(builtins, name):
            name = f"{name}_"

        # increment repeated type names: table, table1, table2, ...
        nth = next(self._shorthand_counters[name]) or ""
        return f"{name}{nth}"

    def render(self, node, code, n_dependents):
        isroot = n_dependents == 0
        ignore = isinstance(node, self.always_ignore)
        assign = n_dependents > 1 or isinstance(node, self.always_assign)

        # depending on the conditions return with (output code, node result) pairs
        if not code:
            return (None, None)
        elif isroot:
            if self.assign_result_to:
                out = f"\n{self.assign_result_to} = {code}\n"
            else:
                out = str(code)
            return (out, code)
        elif ignore:
            return (None, code)
        elif assign:
            var = self.variable_for(node)
            out = f"{var} = {code}\n"
            return (out, var)
        else:
            return (None, code)


@experimental
def decompile(
    expr: ir.Expr,
    render_import: bool = True,
    assign_result_to: str = "result",
    format: bool = False,
) -> str:
    """Decompile an ibis expression into Python source code.

    Parameters
    ----------
    expr
        node or expression to decompile
    render_import
        Whether to add `import ibis` to the result.
    assign_result_to
        Variable name to store the result at, pass None to avoid assignment.
    format
        Whether to format the generated code using black code formatter.

    Returns
    -------
    str
        Equivalent Python source code for `node`.

    """
    if not isinstance(expr, ir.Expr):
        raise TypeError(f"Expected ibis expression, got {type(expr).__name__}")

    node = expr.op()
    node = simplify(node)
    out = io.StringIO()
    ctx = CodeContext(assign_result_to=assign_result_to)
    graph = Graph(node)
    dependents = graph.invert()

    # a relation joined through a `JoinReference` is referenced by the join's
    # predicates and projections, so it must be bound to a name
    joined = collections.Counter()
    for n in graph:
        if isinstance(n, ops.JoinReference):
            joined[n.parent] += len(dependents[n])
        # ``(pred).any()`` resolves the outer table of an EXISTS from the
        # relation whose predicates or values hold it
        if isinstance(n, ops.Filter):
            values = n.predicates
        elif isinstance(n, ops.Project):
            values = n.values.values()
        else:
            continue
        for value in values:
            for sub in value.find(ops.ExistsSubquery, filter=ops.Value):
                ctx.exists_outer[sub] = n.parent

    def fn(node, _, *args, **kwargs):
        if isinstance(node, ctx.needs_rendered):
            kwargs["rendered"] = ctx.rendered
        if isinstance(node, ops.ExistsSubquery):
            kwargs["outer"] = ctx.exists_outer.get(node)
        code = translate(node, *args, **kwargs)
        n_dependents = len(dependents[node]) + joined[node]

        code, result = ctx.render(node, code, n_dependents)
        if code:
            out.write(code)
        ctx.rendered[node] = result

        return result

    node.map(fn)

    result = out.getvalue()
    if render_import:
        result = f"import ibis\n\n\n{result}"

    if format:
        try:
            import black
        except ImportError:
            raise ImportError(
                "The 'format' option requires the 'black' package to be installed"
            )

        result = black.format_str(result, mode=black.FileMode())

    return result
