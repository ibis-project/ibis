from __future__ import annotations

import functools
import itertools
import textwrap
import types
from collections.abc import Mapping, Sequence

from public import public

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util

_infix_ops = {
    # comparison operations
    ops.Equals: "==",
    ops.IdenticalTo: "===",
    ops.NotEquals: "!=",
    ops.Less: "<",
    ops.LessEqual: "<=",
    ops.Greater: ">",
    ops.GreaterEqual: ">=",
    # arithmetic operations
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.FloorDivide: "//",
    ops.Modulus: "%",
    ops.Power: "**",
    # temporal operations
    ops.DateAdd: "+",
    ops.DateSub: "-",
    ops.DateDiff: "-",
    ops.TimeAdd: "+",
    ops.TimeSub: "-",
    ops.TimeDiff: "-",
    ops.TimestampAdd: "+",
    ops.TimestampSub: "-",
    ops.TimestampDiff: "-",
    ops.IntervalAdd: "+",
    ops.IntervalSubtract: "-",
    ops.IntervalMultiply: "*",
    ops.IntervalFloorDivide: "//",
    # boolean operators
    ops.And: "&",
    ops.Or: "|",
    ops.Xor: "^",
}


def type_info(datatype) -> str:
    """Format `datatype` for display next to a column."""
    return f"  # {datatype}" * ibis.options.repr.show_types


def truncate(pieces: Sequence[str], limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    elif limit == 1:
        return pieces[-1:]
    elif limit >= len(pieces):
        return pieces

    first_n = limit // 2
    last_m = limit - first_n
    first, last = pieces[:first_n], pieces[-last_m:]

    maxlen = max(*map(len, first), *map(len, last))
    ellipsis = util.VERTICAL_ELLIPSIS.center(maxlen)

    return [*first, ellipsis, *last]


def render(obj, indent_level=0, limit_items=None, key_separator=":"):
    if isinstance(obj, str):
        result = obj
    elif isinstance(obj, Mapping):
        rendered = {f"{k}{key_separator}": render(v) for k, v in obj.items() if v}
        if not rendered:
            return ""
        maxlen = max(map(len, rendered.keys()))
        lines = [f"{k:<{maxlen}} {v}" for k, v in rendered.items()]
        if limit_items is not None:
            lines = truncate(lines, limit_items)
        result = "\n".join(lines)
    elif isinstance(obj, Sequence):
        lines = tuple(render(item) for item in obj)
        if limit_items is not None:
            lines = truncate(lines, limit_items)
        result = "\n".join(lines)
    else:
        result = str(obj)

    return util.indent(result, spaces=indent_level * 2)


def render_fields(fields, indent_level=0, limit_items=None):
    rendered = {k: render(v, 1) for k, v in fields.items() if v}
    lines = [f"{k}:\n{v}" for k, v in rendered.items()]
    if limit_items is not None:
        lines = truncate(lines, limit_items)
    result = "\n".join(lines)
    return util.indent(result, spaces=indent_level * 2)


def render_schema(schema, indent_level=0, limit_items=None):
    if not len(schema):
        return util.indent("<empty schema>", spaces=indent_level * 2)
    if limit_items is None:
        limit_items = ibis.options.repr.table_columns
    return render(schema, indent_level, limit_items, key_separator="")


def inline(obj):
    if isinstance(obj, Mapping):
        fields = ", ".join(f"{k!r}: {inline(v)}" for k, v in obj.items())
        return f"{{{fields}}}"
    elif util.is_iterable(obj):
        elems = ", ".join(inline(item) for item in obj)
        return f"[{elems}]"
    elif isinstance(obj, types.FunctionType):
        return obj.__name__
    elif isinstance(obj, dt.DataType):
        return str(obj)
    else:
        return repr(obj)


def inline_args(fields, prefer_positional=False):
    fields = {k: inline(v) for k, v in fields.items() if v}

    if fields and prefer_positional:
        first, *rest = fields.keys()
        if not rest:
            return fields[first]
        elif first in {"arg", "expr"}:
            first = fields[first]
            rest = (f"{k}={fields[k]}" for k in rest)
            return ", ".join((first, *rest))

    return ", ".join(f"{k}={v}" for k, v in fields.items())


class Rendered(str):
    def __repr__(self):
        return self


@public
def pretty(node):
    if isinstance(node, ir.Expr):
        node = node.op()
    elif not isinstance(node, ops.Node):
        raise TypeError(f"Expected an expression , got {type(node)}")

    refcnt = itertools.count()
    tables = {}

    def mapper(op, _, **kwargs):
        result = fmt(op, **kwargs)
        if isinstance(op, ops.Relation):
            tables[op] = result
            result = f"r{next(refcnt)}"
        return Rendered(result)

    results = node.map(mapper)

    out = []
    for table, rendered in tables.items():
        if table is not node:
            ref = results[table]
            out.append(f"{ref} := {rendered}")

    res = results[node]
    if isinstance(node, ops.Literal):
        out.append(res)
    elif isinstance(node, ops.Value):
        out.append(f"{node.name}: {res}{type_info(node.dtype)}")
    elif isinstance(node, ops.Relation):
        out.append(tables[node])

    return "\n\n".join(out)


@functools.singledispatch
def fmt(op, **kwargs):
    raise NotImplementedError(f"no pretty printer for {type(op)}")


@fmt.register(ops.Relation)
@fmt.register(ops.DummyTable)
@fmt.register(ops.WindowingTVF)
def _relation(op, **kwargs):
    schema = render_schema(op.schema, indent_level=1)
    return f"{op.__class__.__name__}\n{schema}"


@fmt.register(ops.PhysicalTable)
def _physical_table(op, name, **kwargs):
    schema = render_schema(op.schema, indent_level=1)
    return f"{op.__class__.__name__}: {name}\n{schema}"


@fmt.register(ops.InMemoryTable)
def _in_memory_table(op, data, **kwargs):
    import rich.pretty

    name = f"{op.__class__.__name__}\n"
    data = rich.pretty.pretty_repr(op.data, max_length=ibis.options.repr.table_columns)
    return name + render_fields({"data": data}, 1)


@fmt.register(ops.SQLQueryResult)
@fmt.register(ops.SQLStringView)
def _sql_query_result(op, query, **kwargs):
    clsname = op.__class__.__name__
    if isinstance(op, ops.SQLStringView):
        child, name = kwargs["child"], kwargs["name"]
        top = f"{clsname}[{child}]: {name}\n"
    else:
        top = f"{clsname}\n"

    query = textwrap.shorten(
        query,
        width=ibis.options.repr.query_text_length,
        placeholder=f" {util.HORIZONTAL_ELLIPSIS}",
    )
    schema = render_schema(op.schema)
    return top + render_fields({"query": query, "schema": schema}, 1)


@fmt.register(ops.FillNa)
@fmt.register(ops.DropNa)
def _fill_na(op, table, **kwargs):
    name = f"{op.__class__.__name__}[{table}]\n"
    return name + render_fields(kwargs, 1)


@fmt.register(ops.Aggregation)
def _aggregation(op, table, **kwargs):
    name = f"{op.__class__.__name__}[{table}]\n"
    kwargs["by"] = {node.name: r for node, r in zip(op.by, kwargs["by"])}
    kwargs["metrics"] = {node.name: r for node, r in zip(op.metrics, kwargs["metrics"])}
    return name + render_fields(kwargs, 1)


@fmt.register(ops.Selection)
def _selection(op, table, selections, **kwargs):
    name = f"{op.__class__.__name__}[{table}]\n"

    # special handling required to support both relation and value selections
    rels, values = [], {}
    for node, rendered in zip(op.selections, selections):
        if isinstance(node, ops.Relation):
            rels.append(rendered)
        else:
            values[node.name] = f"{rendered}{type_info(node.dtype)}"

    segments = filter(None, [render(rels), render(values)])
    kwargs["selections"] = "\n".join(segments)

    return name + render_fields(kwargs, 1)


@fmt.register(ops.SetOp)
def _set_op(op, left, right, distinct):
    args = [str(left), str(right)]
    if op.distinct is not None:
        args.append(f"distinct={distinct}")
    return f"{op.__class__.__name__}[{', '.join(args)}]"


@fmt.register(ops.Join)
def _join(op, left, right, predicates, **kwargs):
    args = [str(left), str(right)]
    name = f"{op.__class__.__name__}[{', '.join(args)}]"

    if len(predicates) == 1:
        # if only one, put it directly after the join on the same line
        top = f"{name} {predicates[0]}"
        fields = kwargs
    else:
        top = f"{name}"
        fields = {"predicates": predicates, **kwargs}

    fields = render_fields(fields, 1)
    return f"{top}\n{fields}" if fields else top


@fmt.register(ops.Limit)
@fmt.register(ops.Sample)
def _limit(op, table, **kwargs):
    params = inline_args(kwargs)
    return f"{op.__class__.__name__}[{table}, {params}]"


@fmt.register(ops.SelfReference)
@fmt.register(ops.Distinct)
def _self_reference(op, table, **kwargs):
    return f"{op.__class__.__name__}[{table}]"


@fmt.register(ops.Literal)
def _literal(op, value, **kwargs):
    if op.dtype.is_interval():
        return f"{value!r} {op.dtype.unit.short}"
    else:
        return f"{value!r}"


@fmt.register(ops.TableColumn)
def _table_column(op, table, name):
    return f"{table}.{name}"


@fmt.register(ops.Value)
def _value(op, **kwargs):
    fields = inline_args(kwargs, prefer_positional=True)
    return f"{op.__class__.__name__}({fields})"


@fmt.register(ops.Alias)
def _alias(op, arg, name):
    return arg


@fmt.register(ops.Binary)
def _binary(op, left, right):
    try:
        symbol = _infix_ops[op.__class__]
    except KeyError:
        return f"{op.__class__.__name__}({left}, {right})"
    else:
        return f"{left} {symbol} {right}"


@fmt.register(ops.ScalarParameter)
def _scalar_parameter(op, dtype, **kwargs):
    return f"$({dtype})"


@fmt.register(ops.SortKey)
def _sort_key(op, expr, **kwargs):
    return f"{'asc' if op.ascending else 'desc'} {expr}"


@fmt.register(ops.GeoSpatialBinOp)
def _geo_bin_op(op, left, right, **kwargs):
    fields = [left, right, inline_args(kwargs)]
    args = ", ".join(f"{field}" for field in fields if field)
    return f"{op.__class__.__name__}({args})"
