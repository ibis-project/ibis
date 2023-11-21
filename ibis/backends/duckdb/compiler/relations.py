from __future__ import annotations

import functools
from collections.abc import Mapping

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import FALSE, NULL, STAR


@functools.singledispatch
def translate_rel(op: ops.TableNode, **_):
    """Translate a table node into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_rel.register(ops.DummyTable)
def _dummy(op: ops.DummyTable, *, values, **_):
    return sg.select(*values)


@translate_rel.register(ops.UnboundTable)
@translate_rel.register(ops.InMemoryTable)
def _physical_table(op, **_):
    return sg.expressions.Table(this=sg.to_identifier(op.name, quoted=True))


@translate_rel.register(ops.DatabaseTable)
def _database_table(op, *, name, namespace, **_):
    return sg.table(name, db=namespace.schema, catalog=namespace.database)


@translate_rel.register(ops.SelfReference)
def _self_ref(op: ops.SelfReference, *, parent, **_):
    return parent.as_(op.name)


@translate_rel.register(ops.JoinChain)
def _join_chain(op: ops.JoinChain, *, first, rest, fields):
    result = sg.select(*(value.as_(key) for key, value in fields.items())).from_(first)

    for link in rest:
        if isinstance(link, sg.exp.Alias):
            link = link.this
        result = result.join(link)
    return result


@translate_rel.register(ops.JoinLink)
def _join_link(op: ops.JoinLink, *, how, table, predicates):
    sides = {
        "inner": None,
        "left": "left",
        "right": "right",
        "semi": "left",
        "anti": "left",
        "cross": None,
    }
    kinds = {
        "inner": "inner",
        "left": "outer",
        "right": "outer",
        "semi": "semi",
        "anti": "anti",
        "cross": "cross",
    }
    res = sg.exp.Join(
        this=table,
        side=sides[how],
        kind=kinds[how],
        on=sg.condition(*predicates),
    )
    return res


@translate_rel.register(ops.Project)
def _selection(op: ops.Project, *, parent, values, **_):
    # needs_alias should never be true here in explicitly, but it may get
    # passed via a (recursive) call to translate_val
    return sg.select(*(value.as_(key) for key, value in values.items())).from_(parent)


@translate_rel.register(ops.Aggregate)
def _aggregation(op: ops.Aggregate, *, parent, groups, metrics, **_):
    sel = sg.select(
        *(value.as_(key) for key, value in groups.items()),
        *(value.as_(key) for key, value in metrics.items()),
    ).from_(parent)

    if groups:
        sel = sel.group_by(*map(sg.exp.convert, range(1, len(groups) + 1)))

    return sel


@translate_rel.register(ops.Filter)
def _filter(op: ops.Filter, *, parent, predicates, **_):
    try:
        return parent.where(predicates)
    except AttributeError:
        return sg.select(STAR).from_(parent).where(*predicates)


@translate_rel.register(ops.Sort)
def _sort(op: ops.Sort, *, parent, keys, **_):
    try:
        return parent.order_by(*keys)
    except AttributeError:
        return sg.select(STAR).from_(parent).order_by(*keys)


@translate_rel.register
def _query(op: ops.SQLQueryResult, *, query, **_):
    return sg.parse_one(query, read="duckdb").subquery()


_SET_OP_FUNC = {
    ops.Union: sg.union,
    ops.Intersection: sg.intersect,
    ops.Difference: sg.except_,
}


# @translate_rel.register
# def _set_op(op: ops.SetOp, *, left, right, **_):
#     if isinstance(left, sg.exp.Table):
#         left = sg.select("*").from_(left)
#
#     if isinstance(right, sg.exp.Table):
#         right = sg.select("*").from_(right)
#
#     return _SET_OP_FUNC[type(op)](
#         left.args.get("this", left),
#         right.args.get("this", right),
#         distinct=op.distinct,
#     )


@translate_rel.register
def _limit(op: ops.Limit, *, parent, n, offset, **_):
    result = sg.select("*").from_(parent)

    if isinstance(n, int):
        result = result.limit(n)
    elif n is not None:
        limit = n
        # TODO: calling `.sql` is a workaround for sqlglot not supporting
        # scalar subqueries in limits
        limit = sg.select(limit).from_(parent).subquery().sql(dialect="duckdb")
        result = result.limit(limit)

    assert offset is not None, "offset is None"

    if not isinstance(offset, int):
        skip = offset
        skip = sg.select(skip).from_(parent).subquery().sql(dialect="duckdb")
    elif not offset:
        return result
    else:
        skip = offset

    return result.offset(skip)


@translate_rel.register
def _distinct(op: ops.Distinct, *, parent, **_):
    return sg.select(STAR).distinct().from_(parent)


@translate_rel.register(ops.DropNa)
def _dropna(op: ops.DropNa, *, parent, how, subset, **_):
    if subset is None:
        subset = [
            sg.column(name, table=parent.alias_or_name)
            for name in op.table.schema.names
        ]

    if subset:
        predicate = functools.reduce(
            sg.and_ if how == "any" else sg.or_,
            (sg.not_(col.is_(NULL)) for col in subset),
        )
    elif how == "all":
        predicate = FALSE
    else:
        predicate = None

    if predicate is None:
        return parent

    try:
        return parent.where(predicate)
    except AttributeError:
        return sg.select(STAR).from_(parent).where(predicate)


@translate_rel.register
def _fillna(op: ops.FillNa, *, parent, replacements, **_):
    if isinstance(replacements, Mapping):
        mapping = replacements
    else:
        mapping = {
            name: replacements for name, dtype in op.schema.items() if dtype.nullable
        }
    exprs = [
        (
            sg.alias(sg.exp.Coalesce(this=sg.column(col), expressions=[alt]), col)
            if (alt := mapping.get(col)) is not None
            else sg.column(col)
        )
        for col in op.schema.keys()
    ]
    return sg.select(*exprs).from_(parent)


# @translate_rel.register
# def _view(op: ops.View, *, child, name: str, **_):
#     # TODO: find a way to do this without creating a temporary view
#     backend = op.child.to_expr()._find_backend()
#     backend._create_temp_view(table_name=name, source=sg.select(STAR).from_(child))
#     return sg.table(name)


# @translate_rel.register
# def _sql_string_view(op: ops.SQLStringView, query: str, **_: Any):
#     table = sg.table(op.name)
#     return sg.select(STAR).from_(table).with_(table, as_=query, dialect="duckdb")
