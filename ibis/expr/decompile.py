from __future__ import annotations

import collections
import functools
import io
import itertools

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.graph import Graph
from ibis.util import experimental

_method_overrides = {
    ops.CountDistinct: "nunique",
    ops.CountStar: "count",
    ops.EndsWith: "endswith",
    ops.ExtractDay: "day",
    ops.ExtractDayOfYear: "day_of_year",
    ops.ExtractEpochSeconds: "epoch_seconds",
    ops.ExtractHour: "hour",
    ops.ExtractMicrosecond: "microsecond",
    ops.ExtractMillisecond: "millisecond",
    ops.ExtractMinute: "minute",
    ops.ExtractMinute: "minute",
    ops.ExtractMonth: "month",
    ops.ExtractQuarter: "quarter",
    ops.ExtractSecond: "second",
    ops.ExtractWeekOfYear: "week_of_year",
    ops.ExtractYear: "year",
    ops.Intersection: "intersect",
    ops.IsNull: "isnull",
    ops.LeftAntiJoin: "anti_join",
    ops.LeftSemiJoin: "semi_join",
    ops.Lowercase: "lower",
    ops.RegexSearch: "re_search",
    ops.SelfReference: "view",
    ops.StartsWith: "startswith",
    ops.StringContains: "contains",
    ops.StringSQLILike: "ilike",
    ops.StringSQLLike: "like",
    ops.TimestampNow: "now",
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


@translate.register(ops.Value)
@translate.register(ops.TableNode)
def value(op, *args, **kwargs):
    method = _get_method_name(op)
    kwargs = [(k, v) for k, v in kwargs.items() if v is not None]

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
        return f"[{', '.join(stmt)}]"


@translate.register(ops.Selection)
def selection(op, table, selections, predicates, sort_keys):
    out = f"{table}"
    if selections:
        out = f"{out}.select({_try_unwrap(selections)})"
    if predicates:
        out = f"{out}.filter({_try_unwrap(predicates)})"
    if sort_keys:
        out = f"{out}.order_by({_try_unwrap(sort_keys)})"
    return out


@translate.register(ops.Aggregation)
def aggregation(op, table, by, metrics, predicates, having, sort_keys):
    out = f"{table}"
    if predicates:
        out = f"{out}.filter({_try_unwrap(predicates)})"
    if by:
        out = f"{out}.group_by({_try_unwrap(by)})"
    if having:
        out = f"{out}.having({_try_unwrap(having)})"
    if metrics:
        out = f"{out}.aggregate({_try_unwrap(metrics)})"
    if sort_keys:
        out = f"{out}.order_by({_try_unwrap(sort_keys)})"
    return out


@translate.register(ops.Join)
def join(op, left, right, predicates):
    method = _get_method_name(op)
    return f"{left}.{method}({right}, {_try_unwrap(predicates)})"


@translate.register(ops.SetOp)
def union(op, left, right, distinct):
    method = _get_method_name(op)
    if distinct:
        return f"{left}.{method}({right}, distinct=True)"
    else:
        return f"{left}.{method}({right})"


@translate.register(ops.Limit)
def limit(op, table, n, offset):
    if offset:
        return f"{table}.limit({n}, {offset})"
    else:
        return f"{table}.limit({n})"


@translate.register(ops.TableColumn)
def table_column(op, table, name):
    return f"{table}.{name}"


@translate.register(ops.SortKey)
def sort_key(op, expr, ascending):
    if ascending:
        return f"{expr}.asc()"
    else:
        return f"{expr}.desc()"


@translate.register(ops.Reduction)
def reduction(op, arg, where, **kwargs):
    method = _get_method_name(op)
    return f"{arg}.{method}()"


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
        return f"ibis.interval({value!r})"
    elif inferred.type() != op.dtype:
        return CallStatement("ibis.literal", f"{value!r}, {dtype}")
    else:
        # prefer plain python literal values if the inferred datatype is the same,
        # though this makes rendering method calls on literals more complicated
        return CallStatement("ibis.literal", repr(value))


@translate.register(ops.Cast)
def cast(op, arg, to):
    return f"{arg}.cast({str(to)!r})"


@translate.register(ops.Between)
def between(op, arg, lower_bound, upper_bound):
    return f"{arg}.between({lower_bound}, {upper_bound})"


@translate.register(ops.IfElse)
def ifelse(op, bool_expr, true_expr, false_null_expr):
    return f"{bool_expr}.ifelse({true_expr}, {false_null_expr})"


@translate.register(ops.SimpleCase)
@translate.register(ops.SearchedCase)
def switch_case(op, cases, results, default, base=None):
    out = f"{base}.case()" if base else "ibis.case()"

    for case, result in zip(cases, results):
        out = f"{out}.when({case}, {result})"

    if default is not None:
        out = f"{out}.else_({default})"

    return f"{out}.end()"


_infix_ops = {
    ops.Equals: "==",
    ops.NotEquals: "!=",
    ops.GreaterEqual: ">=",
    ops.Greater: ">",
    ops.LessEqual: "<=",
    ops.Less: "<",
    ops.And: "and",
    ops.Or: "or",
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.Power: "**",
    ops.Modulus: "%",
    ops.TimestampAdd: "+",
    ops.TimestampSub: "-",
    ops.TimestampDiff: "-",
}


@translate.register(ops.Binary)
def binary(op, left, right):
    operator = _infix_ops[type(op)]
    left = _maybe_add_parens(op.left, left)
    right = _maybe_add_parens(op.right, right)
    return f"{left} {operator} {right}"


class CodeContext:
    always_assign = (ops.ScalarParameter, ops.UnboundTable, ops.Aggregation)
    always_ignore = (ops.TableColumn, dt.Primitive, dt.Variadic, dt.Temporal)
    shorthands = {
        ops.Aggregation: "agg",
        ops.Literal: "lit",
        ops.ScalarParameter: "param",
        ops.Selection: "proj",
        ops.TableNode: "t",
    }

    def __init__(self, assign_result_to="result"):
        self.assign_result_to = assign_result_to
        self._shorthand_counters = collections.defaultdict(itertools.count)

    def variable_for(self, node):
        klass = type(node)
        if isinstance(node, ops.TableNode) and isinstance(node, ops.Named):
            name = node.name
        elif klass in self.shorthands:
            name = self.shorthands[klass]
        else:
            name = klass.__name__.lower()

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
    node: ops.Node | ir.Expr,
    render_import: bool = True,
    assign_result_to: str = "result",
    format: bool = False,
) -> str:
    """Decompile an ibis expression into Python source code.

    Parameters
    ----------
    node
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
    if isinstance(node, ir.Expr):
        node = node.op()
    elif not isinstance(node, ops.Node):
        raise TypeError(
            f"Expected ibis expression or operation, got {type(node).__name__}"
        )

    out = io.StringIO()
    ctx = CodeContext(assign_result_to=assign_result_to)
    dependents = Graph(node).invert()

    def fn(node, _, *args, **kwargs):
        code = translate(node, *args, **kwargs)
        n_dependents = len(dependents[node])

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
