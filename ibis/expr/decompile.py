from __future__ import annotations

import builtins
import collections
import enum
import functools
import inspect
import io
import itertools
import keyword

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.graph import Graph
from ibis.expr.rewrites import flatten_predicates, simplify
from ibis.util import experimental

_method_overrides = {
    ops.ApproxCountDistinct: "approx_nunique",
    ops.ArrayCollect: "collect",
    ops.CountDistinct: "nunique",
    ops.CountStar: "count",
    ops.DateNow: "today",
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
    ops.TimestampNow: "now",
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
        if raw == params[name].default:
            continue
        if isinstance(raw, (tuple, list)):
            items = (
                r if isinstance(x, ops.Node) else _render_raw(x)
                for x, r in zip(raw, rendered)
            )
            rendered = f"[{_inline(items)}]"
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

    # if there is a single keyword argument prefer to pass that as positional,
    # unless an earlier optional parameter was left out
    if not args and len(kwargs) == 1 and kwargs[0][0] == op.__argnames__[1]:
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


_prefix_ops = {ops.Not: "~", ops.Negate: "-"}


@translate.register(ops.Not)
@translate.register(ops.Negate)
def unary(op, arg):
    # parenthesized like a binary operator: ``~x.name('y')`` would apply the
    # method first; ``str`` keeps a literal operand an ibis expression
    operator = _prefix_ops[type(op)]
    return f"({operator}{_maybe_add_parens(op.arg, str(arg))})"


def _resolves_deferred(parent, child):
    """Whether `parent` still rebuilds when `child` is spelled as a Deferred."""
    # operators and methods called on a Deferred give a Deferred the table
    # method resolves, and so does a reduction's ``where``; other methods
    # reject one as an argument
    return (
        isinstance(parent, ops.Binary)
        or parent.__args__[0] == child
        or (isinstance(parent, ops.Reduction) and parent.where == child)
    )


def _exists_outer(node, dependents):
    """Return the relation ``.any()`` would evaluate EXISTS `node` against.

    None if there is no single such relation, or if a value holding `node`
    cannot take the Deferred ``.any()`` builds.
    """
    outers = set()
    stack = [node]
    while stack:
        child = stack.pop()
        for parent in dependents[child]:
            if isinstance(parent, ops.Relation) and hasattr(parent, "parent"):
                outers.add(parent.parent)
            elif isinstance(parent, ops.Value) and _resolves_deferred(parent, child):
                stack.append(parent)
            else:
                return None
    return outers.pop() if len(outers) == 1 else None


@translate.register(ops.ExistsSubquery)
def exists_subquery(op, rel, rendered, outer):
    # ``(pred).any()`` is the documented way to spell a correlated EXISTS. It
    # takes the outer table from the relation it is evaluated against, so it
    # rebuilds only when the predicates span exactly the inner relation and
    # that outer one, and hold no EXISTS of their own; ``.any()`` builds a
    # single predicate, so the inner ones are split into their conjuncts and
    # rendered alike whether they were one AND or several predicates
    if isinstance(op.rel, ops.Filter) and outer is not None:
        preds = [
            conjunct
            for pred in op.rel.predicates
            for conjunct in flatten_predicates(pred)
        ]
        relations = set().union(*(pred.relations for pred in preds))
        if (
            len(relations) == 2
            and relations == {op.rel.parent, outer}
            and not any(
                pred.find(ops.ExistsSubquery, filter=ops.Value) for pred in preds
            )
        ):
            pieces = [_maybe_add_parens(pred, str(rendered[pred])) for pred in preds]
            return f"({' & '.join(pieces)}).any()"
    # any other EXISTS has no public spelling; ``rel.count() > 0`` would build a
    # scalar subquery instead
    return f"ibis.expr.operations.ExistsSubquery({rel}.op()).to_expr()"


def _column(rel, name):
    """Render column `name` of the rendered relation `rel`."""
    if (
        name.isidentifier()
        and not keyword.iskeyword(name)
        and not hasattr(ir.Table, name)
    ):
        return f"{rel}.{name}"
    return f"{rel}[{name!r}]"


def _subquery_value(op):
    """Return the value a scalar or IN subquery renders in place of its relation.

    ``as_scalar`` and ``isin`` project a column of another relation themselves,
    and ``isin`` a derived column too, so a relation selecting just that value
    is not rendered.
    """
    rel = op.rel
    if isinstance(rel, ops.Project) and len(rel.values) == 1:
        ((name, value),) = rel.values.items()
        if isinstance(value, ops.Field):
            if value.rel == rel.parent and value.name == name:
                return value
        elif isinstance(op, ops.InSubquery) and value.shape.is_columnar():
            return value
    return None


def _subquery_column(op, rendered):
    """Render the value of subquery `op` in place of its relation."""
    value = _subquery_value(op)
    if isinstance(value, ops.Field):
        return _column(rendered[value.rel], value.name)
    (name,) = op.rel.values
    (column,) = _wrap_alias(op.rel.values, {name: rendered[value]})
    return column


@translate.register(ops.ScalarSubquery)
def scalar_subquery(op, rel, rendered):
    if _subquery_value(op) is None:
        return f"{rel}.as_scalar()"
    return f"{_subquery_column(op, rendered)}.as_scalar()"


@translate.register(ops.InSubquery)
def in_subquery(op, rel, needle, rendered):
    if _subquery_value(op) is None:
        (name,) = op.rel.schema.names
        return f"{needle}.isin({_column(rel, name)})"
    return f"{needle}.isin({_subquery_column(op, rendered)})"


@translate.register(ops.NTile)
def ntile(op, buckets):
    return f"ibis.ntile({buckets})"


@translate.register(ops.WindowBoundary)
def window_boundary(op, value, preceding):
    if isinstance(value, CallStatement):
        # ``-0`` is ``0``, which rebuilds as a following boundary, so a
        # preceding zero stays an ibis literal to carry the sign
        zero = getattr(op.value, "value", None) == 0
        value = str(value) if preceding and zero else value.args
    if preceding:
        return f"-{value}"
    return value


@translate.register(ops.WindowFunction)
def window_function(op, func, how, start, end, group_by, order_by):
    params = []
    if group_by:
        params.append(f"group_by=[{_inline(group_by)}]")
    if order_by:
        params.append(f"order_by=[{_inline(order_by)}]")
    if _negative_bound(op.start) or _negative_bound(op.end):
        # ``rows=``/``range=`` read a negative bound as preceding; only the
        # two-sided builders keep its side
        builder = "window" if op.how == "rows" else "range_window"
        bounds = [
            f"{side}={bound.value.value!r}"
            for side, bound in (("preceding", op.start), ("following", op.end))
            if bound is not None
        ]
        return f"{func}.over(ibis.{builder}({', '.join([*bounds, *params])}))"
    if start is not None or end is not None or op.how == "range":
        params.append(f"{how}=({start}, {end})")
    return f"{func}.over({', '.join(params)})"


def _negative_bound(bound):
    value = getattr(getattr(bound, "value", None), "value", None)
    return isinstance(value, (int, float)) and value < 0


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
    if not hasattr(ir.Table, f"{how}_join"):
        # there is no ``positional_join``
        return f".join({table}, how={how!r})"
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


@translate.register(ops.Set)
def union(op, left, right, distinct):
    method = _get_method_name(op)
    # the default of ``distinct`` differs between the set operation methods
    params = inspect.signature(getattr(ir.Table, method)).parameters
    if op.distinct != params["distinct"].default:
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
@translate.register(ops.DateNow)
@translate.register(ops.RankBase)
@translate.register(ops.TimestampNow)
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
    elif isinstance(op.dtype, dt.Interval) and value is not None:
        return f"ibis.interval({value!r}, {op.dtype.unit.value!r})"
    elif isinstance(op.dtype, dt.Decimal) and value is not None:
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
    # a plain Python value on the left would fold with a plain right operand, or
    # swap places with an ibis one through the reflected operator; ``str`` keeps
    # it an ibis expression
    left = _maybe_add_parens(op.left, str(left))
    right = _maybe_add_parens(op.right, right)
    return _maybe_add_parens(op, f"{left} {operator} {right}")


@translate.register(ops.InValues)
def isin(op, value, options):
    return f"{value}.isin([{_inline(options)}])"


@translate.register(ops.DummyTable)
def dummy_table(op, values):
    if len(values) == 1:
        (value,) = _wrap_alias(op.values, values)
        return f"{value}.as_table()"
    # several values have no public spelling; ``mutate`` would build a
    # projection over a one-value table
    items = ", ".join(
        f"{k!r}: {_maybe_add_parens(op.values[k], str(v))}.op()"
        for k, v in values.items()
    )
    return f"ibis.expr.operations.DummyTable({{{items}}}).to_expr()"


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

    def __init__(self, assign_result_to="result"):
        self.assign_result_to = assign_result_to
        self._shorthand_counters = collections.defaultdict(itertools.count)

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
    joined = set()
    # a subquery that renders its single value in place of its relation does
    # not use the relation
    bypassed = collections.Counter()
    exists_outer = {}
    for n in graph:
        if isinstance(n, ops.JoinReference):
            joined.add(n.parent)
        elif isinstance(n, (ops.ScalarSubquery, ops.InSubquery)):
            if _subquery_value(n) is not None:
                bypassed[n.rel] += 1
        elif isinstance(n, ops.ExistsSubquery):
            exists_outer[n] = _exists_outer(n, dependents)

    # positional-only, since op fields arrive as keywords (`SimpleCase.results`)
    def fn(node, results, /, **kwargs):
        if isinstance(node, ops.Subquery):
            kwargs["rendered"] = results
        if isinstance(node, ops.ExistsSubquery):
            kwargs["outer"] = exists_outer[node]
        code = translate(node, **kwargs)
        n_dependents = len(dependents[node])
        if node in bypassed:
            # still one use, so that it renders inline rather than as the root
            n_dependents = max(n_dependents - bypassed[node], 1)
        n_dependents += node in joined

        code, result = ctx.render(node, code, n_dependents)
        if code:
            out.write(code)

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
