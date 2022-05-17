import collections
import functools
import itertools
import sys

from matchpy import (
    CustomConstraint,
    Pattern,
    ReplacementRule,
    Symbol,
    Wildcard,
    create_operation_expression,
    is_match,
    match,
    op_iter,
    op_len,
    replace,
    replace_all,
    replace_many,
    substitute,
)

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.expr.window as win
import ibis.util as util
from ibis.common.graph import Graph

_ = Wildcard.dot()


# after Node.__iter__ and __len__ restored we won't need these
# @op_iter.register(ops.Node)
# def iter_node_operands(node):
#     return iter(node.__args__)


# @op_len.register(ops.Node)
# def iter_node_operands(node):
#     return len(node.__args__)


# could overcome by support list signature for List
@create_operation_expression.register(ops.Node)
def create_operation_node(old_operation, new_operands, variable_name=True):
    return type(old_operation)(*new_operands, bypass_validation=True)


@create_operation_expression.register(ops.List)
def create_operation_node(old_operation, new_operands, variable_name=True):
    return type(old_operation)(new_operands, bypass_validation=True)


class TableRef(ops.TableNode):
    number = rlz.instance_of(int)

    def _assert_valid(self, exprs):
        return True


# variable names in lambda matters!!!!!!!!! must match the variable names from
# the wildcards otherwise they get omitted
# has_schema = Pattern(
#     Wildcard.dot('table'),
#     CustomConstraint(lambda table: isinstance(table, sch.HasSchema)),
# )
# table_ = Pattern(ops.TableNode())
# table_ = Pattern(
#     Wildcard.dot('table'),
#     CustomConstraint(lambda table: isinstance(table, ops.TableNode))
# )
# table_ = Pattern(ops.UnboundTable(_, _))
table_ = Pattern(
    Wildcard.dot('table'),
    CustomConstraint(lambda table: isinstance(table, ops.UnboundTable)),
)


def fmt_truncated(
    pieces,
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


@functools.singledispatch
def fmt(node: ops.Node, **kwargs) -> str:
    raise NotImplementedError(node)


@fmt.register
def fmt_schema(node: sch.Schema, names, types) -> str:
    maxlen = max(map(len, names))
    cols = [f"{name:<{maxlen}} {typ}" for name, typ in zip(names, types)]
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


@fmt.register
def fmt_table_ref(node: TableRef, number) -> str:
    return f"r{number}"


@fmt.register
def fmt_unbound_table(node: ops.UnboundTable, name, schema) -> str:
    return f"{node.__class__.__name__}: {name}\n{schema}"


@fmt.register
def fmt_selection(
    node: ops.Selection, table, selections, predicates, sort_keys
) -> str:
    top = f"{node.__class__.__name__}[{table}]"
    return top
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
    print("s")


def pretty(expr):
    node = expr.op()

    counter = itertools.count()
    refs = collections.defaultdict(lambda: TableRef(next(counter)))
    node = replace_all(
        node,
        [ReplacementRule(table_, lambda table: refs[table])],
    )
    print()
    print(node)

    # pieces = [f"r{ref.number}: {fmt(table)}" for table, ref in refs.items()]
    # pieces += list(map(fmt, g.toposort()))

    g = Graph(node, node_types=(ops.Node, sch.Schema))
    results = g.map(fmt)

    for k, v in results.items():
        print('==============')
        print(v)


# table = ibis.table([("a", dt.int64), ("b", dt.float64)], name="alfa").op()
# tables = ops.List(values=[
#     ibis.table([("a", dt.int64), ("b", dt.float64)], name="alfa"),
#     ibis.table([("a", dt.int64), ("b", dt.float64)], name="alfa"),
#     ibis.table([("a", dt.int64), ("b", dt.float64)], name="beta"),
# ])


table = ibis.table(
    [("col", "int64"), ("col2", "string"), ("col3", "double")],
    name="t",
).mutate(col4=lambda t: t.col2.length())

pretty(table)


# result = repr(table)
# expected = """\
# r0 := UnboundTable: t
#   col  int64
#   col2 string
#   col3 float64

# Selection[r0]
#   selections:
#     r0
#     col4: StringLength(r0.col2)"""
# assert result == expected
