from __future__ import annotations

import collections
import functools
import textwrap
import types  # noqa: TCH003
from typing import Any, Callable, Deque, Iterable, Mapping, Tuple

import rich.pretty

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.common import graph

Aliases = Mapping[ops.TableNode, int]
Deps = Deque[Tuple[int, ops.TableNode]]


class Alias:
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"r{self.value}"


def fmt(expr: ir.Expr) -> str:
    """Format `expr`.

    Main entry point for the `Expr.__repr__` implementation.

    Returns
    -------
    str
        Formatted expression
    """
    *deps, root = graph.toposort(expr.op()).keys()
    deps = collections.deque(
        (Alias(alias), dep)
        for alias, dep in enumerate(
            dep for dep in deps if isinstance(dep, ops.TableNode)
        )
    )

    aliases = {dep: alias for alias, dep in deps}
    pieces = []

    while deps:
        alias, node = deps.popleft()
        formatted = fmt_table_op(node, aliases=aliases, deps=deps)
        pieces.append(f"{alias} := {formatted}")

    name = expr.get_name() if expr.has_name() else None
    pieces.append(fmt_root(root, name=name, aliases=aliases, deps=deps))
    depth = ibis.options.repr.depth or 0
    if depth and depth < len(pieces):
        return fmt_truncated(pieces, depth=depth)
    return "\n\n".join(pieces)


def fmt_truncated(
    pieces: Iterable[str],
    *,
    depth: int,
    sep: str = "\n\n",
    ellipsis: str = util.VERTICAL_ELLIPSIS,
) -> str:
    if depth == 1:
        return pieces[-1]

    first_n = depth // 2
    last_m = depth - first_n
    return sep.join([*pieces[:first_n], ellipsis, *pieces[-last_m:]])


def selection_maxlen(nodes: Iterable[ops.Node]) -> int:
    """Compute the length of the longest name of input expressions."""
    return max(
        (len(node.name) for node in nodes if isinstance(node, ops.Named)), default=0
    )


@functools.singledispatch
def fmt_root(op: ops.Node, *, aliases: Aliases, **_: Any) -> str:
    """Fallback formatting implementation."""
    raw_parts = fmt_fields(
        op,
        dict.fromkeys(op.argnames, fmt_value),
        aliases=aliases,
    )
    return f"{op.__class__.__name__}\n{raw_parts}"


@fmt_root.register
def _fmt_root_table_node(op: ops.TableNode, **kwargs: Any) -> str:
    return fmt_table_op(op, **kwargs)


@fmt_root.register
def _fmt_root_value_op(op: ops.Value, *, name: str, aliases: Aliases, **_: Any) -> str:
    value = fmt_value(op, aliases=aliases)
    prefix = f"{name}: " if name is not None else ""
    return f"{prefix}{value}{type_info(op.to_expr().type())}"


@fmt_root.register
def _fmt_root_literal_op(
    op: ops.Literal, *, name: str, aliases: Aliases, **_: Any
) -> str:
    value = fmt_value(op, aliases=aliases)
    return f"{value}{type_info(op.to_expr().type())}"


@fmt_root.register(ops.SortKey)
def _fmt_root_sort_key(op: ops.SortKey, *, aliases: Aliases, **_: Any) -> str:
    return fmt_value(op, aliases=aliases)


@functools.singledispatch
def fmt_table_op(op: ops.TableNode, **_: Any) -> str:
    raise AssertionError(f'`fmt_table_op` not implemented for operation: {type(op)}')


@fmt_table_op.register
def _fmt_table_op_physical_table(op: ops.PhysicalTable, **_: Any) -> str:
    top = f"{op.__class__.__name__}: {op.name}"
    formatted_schema = fmt_schema(op.schema)
    return f"{top}\n{formatted_schema}"


def fmt_schema(schema: sch.Schema) -> str:
    """Format `schema`.

    Parameters
    ----------
    schema
        Ibis schema to format

    Returns
    -------
    str
        Formatted schema
    """
    names = schema.names
    maxlen = max(map(len, names))
    cols = [f"{name:<{maxlen}} {typ}" for name, typ in schema.items()]
    depth = ibis.options.repr.table_columns
    if depth is not None and depth < len(cols):
        first_column_name = names[0]
        raw = fmt_truncated(
            cols,
            depth=depth,
            sep="\n",
            ellipsis=util.VERTICAL_ELLIPSIS.center(len(first_column_name)),
        )
    else:
        raw = "\n".join(cols)

    return util.indent(raw, spaces=2)


@fmt_table_op.register
def _fmt_table_op_sql_query_result(op: ops.SQLQueryResult, **_: Any) -> str:
    short_query = textwrap.shorten(
        op.query,
        ibis.options.repr.query_text_length,
        placeholder=f" {util.HORIZONTAL_ELLIPSIS}",
    )
    query = f"query: {short_query!r}"
    top = op.__class__.__name__
    formatted_schema = fmt_schema(op.schema)
    schema_field = util.indent(f"schema:\n{formatted_schema}", spaces=2)
    return f"{top}\n{util.indent(query, spaces=2)}\n{schema_field}"


@fmt_table_op.register
def _fmt_table_op_view(op: ops.View, *, aliases: Aliases, **_: Any) -> str:
    top = op.__class__.__name__
    formatted_schema = fmt_schema(op.schema)
    schema_field = util.indent(f"schema:\n{formatted_schema}", spaces=2)
    return f"{top}[{aliases[op.child]}]: {op.name}\n{schema_field}"


@fmt_table_op.register
def _fmt_table_op_sql_view(
    op: ops.SQLStringView,
    *,
    aliases: Aliases,
    **_: Any,
) -> str:
    short_query = textwrap.shorten(
        op.query,
        ibis.options.repr.query_text_length,
        placeholder=f" {util.HORIZONTAL_ELLIPSIS}",
    )
    query = f"query: {short_query!r}"
    top = op.__class__.__name__
    formatted_schema = fmt_schema(op.schema)
    schema_field = util.indent(f"schema:\n{formatted_schema}", spaces=2)
    components = [
        f"{top}[{aliases[op.child]}]: {op.name}",
        util.indent(query, spaces=2),
        schema_field,
    ]
    return "\n".join(components)


@functools.singledispatch
def fmt_join(op: ops.Join, *, aliases: Aliases) -> tuple[str, str]:
    raise AssertionError(f'join type {type(op)} not implemented')


@fmt_join.register(ops.Join)
def _fmt_join(op: ops.Join, *, aliases: Aliases) -> tuple[str, str]:
    # format the operator and its relation inputs
    left = aliases[op.left]
    right = aliases[op.right]
    top = f"{op.__class__.__name__}[{left}, {right}]"

    # format the join predicates
    # if only one, put it directly after the join on the same line
    # if more than one put each on a separate line
    preds = op.predicates
    formatted_preds = [fmt_value(pred, aliases=aliases) for pred in preds]
    has_one_pred = len(preds) == 1
    sep = " " if has_one_pred else "\n"
    joined_predicates = util.indent(
        "\n".join(formatted_preds),
        spaces=2 * (not has_one_pred),
    )
    trailing_sep = "\n" + "\n" * (not has_one_pred)
    return f"{top}{sep}{joined_predicates}", trailing_sep


@fmt_join.register(ops.AsOfJoin)
def _fmt_asof_join(op: ops.AsOfJoin, *, aliases: Aliases) -> tuple[str, str]:
    left = aliases[op.left]
    right = aliases[op.right]
    top = f"{op.__class__.__name__}[{left}, {right}]"
    raw_parts = fmt_fields(
        op,
        dict(predicates=fmt_value, by=fmt_value, tolerance=fmt_value),
        aliases=aliases,
    )
    return f"{top}\n{raw_parts}", "\n\n"


@fmt_table_op.register
def _fmt_table_op_join(
    op: ops.Join,
    *,
    aliases: Aliases,
    deps: Deps,
    **_: Any,
) -> str:
    # first, format the current join operation
    result, join_sep = fmt_join(op, aliases=aliases)
    formatted_joins = [result, join_sep]

    # process until the first non-Join dependency is popped in other words
    # process all runs of joins
    alias, current = None, None
    if deps:
        alias, current = deps.popleft()

        while isinstance(current, ops.Join):
            # copy the alias so that mutations to the value aren't shared
            # format the `current` join
            formatted_join, join_sep = fmt_join(current, aliases=aliases)
            formatted_joins.append(f"{alias} := {formatted_join}")
            formatted_joins.append(join_sep)

            if not deps:
                break

            alias, current = deps.popleft()

        if current is not None and not isinstance(current, ops.Join):
            # the last node popped from `deps` isn't a join which means we
            # still need to process it, so we put it at the front of the queue
            deps.appendleft((alias, current))

    # we don't want the last trailing separator so remove it from the end
    formatted_joins.pop()
    return "".join(formatted_joins)


@fmt_table_op.register
def _(op: ops.CrossJoin, *, aliases: Aliases, **_: Any) -> str:
    left = aliases[op.left]
    right = aliases[op.right]
    return f"{op.__class__.__name__}[{left}, {right}]"


def _fmt_set_op(
    op: ops.SetOp,
    *,
    aliases: Aliases,
    distinct: bool | None = None,
) -> str:
    args = [str(aliases[op.left]), str(aliases[op.right])]
    if distinct is not None:
        args.append(f"distinct={distinct}")
    return f"{op.__class__.__name__}[{', '.join(args)}]"


@fmt_table_op.register
def _fmt_table_op_set_op(op: ops.SetOp, *, aliases: Aliases, **_: Any) -> str:
    return _fmt_set_op(op, aliases=aliases)


@fmt_table_op.register
def _fmt_table_op_union(op: ops.Union, *, aliases: Aliases, **_: Any) -> str:
    return _fmt_set_op(op, aliases=aliases, distinct=op.distinct)


@fmt_table_op.register(ops.SelfReference)
@fmt_table_op.register(ops.Distinct)
def _fmt_table_op_self_reference_distinct(
    op: ops.Distinct | ops.SelfReference,
    *,
    aliases: Aliases,
    **_: Any,
) -> str:
    return f"{op.__class__.__name__}[{aliases[op.table]}]"


@fmt_table_op.register
def _fmt_table_op_fillna(op: ops.FillNa, *, aliases: Aliases, **_: Any) -> str:
    top = f"{op.__class__.__name__}[{aliases[op.table]}]"
    raw_parts = fmt_fields(op, dict(replacements=fmt_value), aliases=aliases)
    return f"{top}\n{raw_parts}"


@fmt_table_op.register
def _fmt_table_op_dropna(op: ops.DropNa, *, aliases: Aliases, **_: Any) -> str:
    top = f"{op.__class__.__name__}[{aliases[op.table]}]"
    how = f"how: {op.how!r}"
    raw_parts = fmt_fields(op, dict(subset=fmt_value), aliases=aliases)
    return f"{top}\n{util.indent(how, spaces=2)}\n{raw_parts}"


def fmt_fields(
    op: ops.TableNode,
    fields: Mapping[str, Callable[[Any, Aliases], str]],
    *,
    aliases: Aliases,
) -> str:
    parts = []

    for field, formatter in fields.items():
        if exprs := [
            expr for expr in util.promote_list(getattr(op, field)) if expr is not None
        ]:
            field_fmt = [formatter(expr, aliases=aliases) for expr in exprs]

            parts.append(f"{field}:")
            parts.append(util.indent("\n".join(field_fmt), spaces=2))

    return util.indent("\n".join(parts), spaces=2)


@fmt_table_op.register
def _fmt_table_op_selection(op: ops.Selection, *, aliases: Aliases, **_: Any) -> str:
    top = f"{op.__class__.__name__}[{aliases[op.table]}]"
    raw_parts = fmt_fields(
        op,
        dict(
            selections=functools.partial(
                fmt_selection_column,
                maxlen=selection_maxlen(op.selections),
            ),
            predicates=fmt_value,
            sort_keys=fmt_value,
        ),
        aliases=aliases,
    )
    return f"{top}\n{raw_parts}"


@fmt_table_op.register
def _fmt_table_op_aggregation(
    op: ops.Aggregation, *, aliases: Aliases, **_: Any
) -> str:
    top = f"{op.__class__.__name__}[{aliases[op.table]}]"
    raw_parts = fmt_fields(
        op,
        dict(
            metrics=functools.partial(
                fmt_selection_column,
                maxlen=selection_maxlen(op.metrics),
            ),
            by=functools.partial(
                fmt_selection_column,
                maxlen=selection_maxlen(op.by),
            ),
            having=fmt_value,
            predicates=fmt_value,
            sort_keys=fmt_value,
        ),
        aliases=aliases,
    )
    return f"{top}\n{raw_parts}"


@fmt_table_op.register
def _fmt_table_op_limit(op: ops.Limit, *, aliases: Aliases, **_: Any) -> str:
    params = [str(aliases[op.table]), f"n={op.n:d}"]
    if offset := op.offset:
        params.append(f"offset={offset:d}")
    return f"{op.__class__.__name__}[{', '.join(params)}]"


@fmt_table_op.register
def _fmt_table_op_in_memory_table(op: ops.InMemoryTable, **_: Any) -> str:
    # arbitrary limit, but some value is needed to avoid a huge repr
    max_length = 10
    pretty_data = rich.pretty.pretty_repr(op.data, max_length=max_length)
    return "\n".join(
        [
            op.__class__.__name__,
            util.indent("data:", spaces=2),
            util.indent(pretty_data, spaces=4),
        ]
    )


@fmt_table_op.register
def _fmt_table_op_dummy_table(op: ops.DummyTable, **_: Any) -> str:
    formatted_schema = fmt_schema(op.schema)
    schema_field = util.indent(f"schema:\n{formatted_schema}", spaces=2)
    return f"{op.__class__.__name__}\n{schema_field}"


@functools.singledispatch
def fmt_selection_column(value_expr: object, **_: Any) -> str:
    raise AssertionError(
        f'expression type not implemented for fmt_selection_column: {type(value_expr)}'
    )


def type_info(datatype: dt.DataType) -> str:
    """Format `datatype` for display next to a column."""
    return f"  # {datatype}" * ibis.options.repr.show_types


@fmt_selection_column.register
def _fmt_selection_column_sequence(node: tuple, **kwargs):
    return "\n".join(fmt_selection_column(value, **kwargs) for value in node.values)


@fmt_selection_column.register
def _fmt_selection_column_value_expr(
    node: ops.Value, *, aliases: Aliases, maxlen: int = 0
) -> str:
    name = f"{node.name}:"
    # the additional 1 is for the colon
    aligned_name = f"{name:<{maxlen + 1}}"
    value = fmt_value(node, aliases=aliases)
    dtype = type_info(node.output_dtype)
    return f"{aligned_name} {value}{dtype}"


@fmt_selection_column.register
def _fmt_selection_column_table_expr(
    node: ops.TableNode, *, aliases: Aliases, **_: Any
) -> str:
    return str(aliases[node])


_BIN_OP_CHARS = {
    # comparison operations
    ops.Equals: "==",
    ops.IdenticalTo: "===",
    ops.NotEquals: "!=",
    ops.Less: "<",
    ops.LessEqual: "<=",
    ops.Greater: ">",
    ops.GreaterEqual: ">=",
    # arithmetic
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


@functools.singledispatch
def fmt_value(obj, **_: Any) -> str:
    """Format a value expression or operation.

    [`repr`][repr] the object if we don't have a specific formatting
    rule.
    """
    return repr(obj)


@fmt_value.register
def _fmt_value_function_type(func: types.FunctionType, **_: Any) -> str:
    return func.__name__


@fmt_value.register
def _fmt_value_node(op: ops.Node, **_: Any) -> str:
    raise AssertionError(f'`fmt_value` not implemented for operation: {type(op)}')


@fmt_value.register
def _fmt_value_sequence(op: tuple, **kwargs: Any) -> str:
    return ", ".join([fmt_value(value, **kwargs) for value in op])


@fmt_value.register
def _fmt_value_expr(op: ops.Value, *, aliases: Aliases) -> str:
    """Format a value expression.

    Forwards the call on to the specific operation dispatch rule.
    """
    return fmt_value(op, aliases=aliases)


@fmt_value.register
def _fmt_value_binary_op(op: ops.Binary, *, aliases: Aliases) -> str:
    left = fmt_value(op.left, aliases=aliases)
    right = fmt_value(op.right, aliases=aliases)
    try:
        op_char = _BIN_OP_CHARS[type(op)]
    except KeyError:
        return f"{type(op).__name__}({left}, {right})"
    else:
        return f"{left} {op_char} {right}"


@fmt_value.register
def _fmt_value_negate(op: ops.Negate, *, aliases: Aliases) -> str:
    op_name = "Not" if op.output_dtype.is_boolean() else "Negate"
    operand = fmt_value(op.arg, aliases=aliases)
    return f"{op_name}({operand})"


@fmt_value.register
def _fmt_value_literal(op: ops.Literal, **_: Any) -> str:
    if op.dtype.is_interval():
        return f"{op.value} {op.dtype.unit.short}"
    return repr(op.value)


@fmt_value.register
def _fmt_value_datatype(datatype: dt.DataType, **_: Any) -> str:
    return str(datatype)


@fmt_value.register
def _fmt_value_value_op(op: ops.Value, *, aliases: Aliases) -> str:
    args = []
    # loop over argument names and original expression
    for argname, orig_expr in zip(op.argnames, op.args):
        # promote argument to a list, so that we don't accidentally repr
        # entire subtrees when all we want is the formatted argument value
        if exprs := [expr for expr in util.promote_list(orig_expr) if expr is not None]:
            # format the individual argument values
            formatted_args = ", ".join(
                fmt_value(expr, aliases=aliases) for expr in exprs
            )
            # if the original argument was a non-string iterable, display it as
            # a list
            value = (
                f"[{formatted_args}]" if util.is_iterable(orig_expr) else formatted_args
            )
            # `arg` and `expr` are noisy, so we ignore printing them as a
            # special case
            if argname not in ("arg", "expr"):
                formatted = f"{argname}={value}"
            else:
                formatted = value
            args.append(formatted)

    return f"{op.__class__.__name__}({', '.join(args)})"


@fmt_value.register
def _fmt_value_alias(op: ops.Alias, *, aliases: Aliases) -> str:
    return fmt_value(op.arg, aliases=aliases)


@fmt_value.register
def _fmt_value_table_column(op: ops.TableColumn, *, aliases: Aliases) -> str:
    return f"{aliases[op.table]}.{op.name}"


@fmt_value.register
def _fmt_value_scalar_parameter(op: ops.ScalarParameter, **_: Any) -> str:
    return f"$({op.dtype})"


@fmt_value.register
def _fmt_value_sort_key(op: ops.SortKey, *, aliases: Aliases) -> str:
    expr = fmt_value(op.expr, aliases=aliases)
    prefix = "asc" if op.ascending else "desc"
    return f"{prefix} {expr}"


@fmt_value.register
def _fmt_value_physical_table(op: ops.PhysicalTable, **_: Any) -> str:
    """Format a table as value.

    This function is called when a table is used in a value expression.
    An example is `table.count()`.
    """
    return op.name


@fmt_value.register
def _fmt_value_table_node(op: ops.TableNode, *, aliases: Aliases, **_: Any) -> str:
    """Format a table as value.

    This function is called when a table is used in a value expression.
    An example is `table.count()`.
    """
    return f"{aliases[op]}"


@fmt_value.register
def _fmt_value_string_sql_like(op: ops.StringSQLLike, *, aliases: Aliases) -> str:
    expr = fmt_value(op.arg, aliases=aliases)
    pattern = fmt_value(op.pattern, aliases=aliases)
    prefix = "I" * isinstance(op, ops.StringSQLILike)
    return f"{expr} {prefix}LIKE {pattern}"


@fmt_value.register
def _fmt_value_window(win: ops.WindowFrame, *, aliases: Aliases) -> str:
    args = []
    for field, value in (
        ("group_by", win.group_by),
        ("order_by", win.order_by),
        ("start", win.start),
        ("end", win.end),
    ):
        disp_field = field.lstrip("_")
        if value is not None:
            if isinstance(value, tuple):
                # don't show empty sequences
                if not value:
                    continue
                elements = ", ".join(
                    fmt_value(
                        arg.op() if isinstance(arg, ir.Expr) else arg,
                        aliases=aliases,
                    )
                    for arg in value
                )
                formatted = f"[{elements}]"
            else:
                formatted = fmt_value(value, aliases=aliases)
            args.append(f"{disp_field}={formatted}")
    return f"{win.__class__.__name__}({', '.join(args)})"
