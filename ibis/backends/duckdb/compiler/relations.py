from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import FALSE, NULL, STAR, lit


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
    try:
        db, catalog = namespace.split(".")
    except AttributeError:
        db = catalog = None
    return sg.table(name, db=db, catalog=catalog)


def replace_tables_with_star_selection(node, alias=None):
    if isinstance(node, (sg.exp.Subquery, sg.exp.Table, sg.exp.CTE)):
        return sg.exp.Column(
            this=STAR,
            table=sg.to_identifier(alias if alias is not None else node.alias_or_name),
        )
    return node


@translate_rel.register
def _selection(op: ops.Selection, *, table, selections, predicates, sort_keys, **_):
    # needs_alias should never be true here in explicitly, but it may get
    # passed via a (recursive) call to translate_val
    if isinstance(op.table, ops.Join) and not isinstance(
        op.table, (ops.LeftSemiJoin, ops.LeftAntiJoin)
    ):
        args = table.this.args
        from_ = args["from"]
        (join,) = args["joins"]
    else:
        from_ = join = None

    selections = tuple(
        replace_tables_with_star_selection(
            node,
            # replace the table name with the alias if the table is **not** a
            # join, because we may be selecting from a subquery or an aliased
            # table; otherwise we'll select from the _unaliased_ table or the
            # _child_ table, which may have a different alias than the one we
            # generated for the input table
            table.alias_or_name if from_ is None and join is None else None,
        )
        for node in selections
    ) or (STAR,)

    sel = sg.select(*selections).from_(from_ if from_ is not None else table)

    if join is not None:
        sel = sel.join(join)

    if predicates:
        if join is not None:
            sel = sg.select(STAR).from_(sel.subquery(table.alias_or_name))
        sel = sel.where(*predicates)

    if sort_keys:
        sel = sel.order_by(*sort_keys)

    return sel


@translate_rel.register(ops.Aggregation)
def _aggregation(
    op: ops.Aggregation, *, table, metrics, by, having, predicates, sort_keys, **_
):
    selections = (by + metrics) or (STAR,)
    sel = sg.select(*selections).from_(table)

    if by:
        sel = sel.group_by(*map(lit, range(1, len(by) + 1)))

    if predicates:
        sel = sel.where(*predicates)

    if having:
        sel = sel.having(*having)

    if sort_keys:
        sel = sel.order_by(*sort_keys)

    return sel


_JOIN_TYPES = {
    ops.InnerJoin: "INNER",
    ops.LeftJoin: "LEFT",
    ops.RightJoin: "RIGHT",
    ops.OuterJoin: "FULL",
    ops.CrossJoin: "CROSS",
    ops.LeftSemiJoin: "SEMI",
    ops.LeftAntiJoin: "ANTI",
    ops.AsOfJoin: "ASOF",
}


@translate_rel.register
def _join(op: ops.Join, *, left, right, predicates, **_):
    on = sg.and_(*predicates) if predicates else None

    join_type = _JOIN_TYPES[type(op)]
    try:
        return left.join(right, join_type=join_type, on=on)
    except AttributeError:
        select_args = [f"{left.alias_or_name}.*"]

        # select from both the left and right side of the join if the join
        # is not a filtering join (semi join or anti join); filtering joins
        # only return the left side columns
        if not isinstance(op, (ops.LeftSemiJoin, ops.LeftAntiJoin)):
            select_args.append(f"{right.alias_or_name}.*")
        return (
            sg.select(*select_args).from_(left).join(right, join_type=join_type, on=on)
        )


@translate_rel.register
def _self_ref(op: ops.SelfReference, *, table, **_):
    return sg.alias(table, op.name)


@translate_rel.register
def _query(op: ops.SQLQueryResult, *, query, **_):
    return sg.parse_one(query, read="duckdb").subquery()


_SET_OP_FUNC = {
    ops.Union: sg.union,
    ops.Intersection: sg.intersect,
    ops.Difference: sg.except_,
}


@translate_rel.register
def _set_op(op: ops.SetOp, *, left, right, **_):
    if isinstance(left, sg.exp.Table):
        left = sg.select("*").from_(left)

    if isinstance(right, sg.exp.Table):
        right = sg.select("*").from_(right)

    return _SET_OP_FUNC[type(op)](
        left.args.get("this", left),
        right.args.get("this", right),
        distinct=op.distinct,
    )


@translate_rel.register
def _limit(op: ops.Limit, *, table, n, offset, **_):
    result = sg.select("*").from_(table)

    if isinstance(n, int):
        result = result.limit(n)
    elif n is not None:
        limit = n
        # TODO: calling `.sql` is a workaround for sqlglot not supporting
        # scalar subqueries in limits
        limit = sg.select(limit).from_(table).subquery().sql(dialect="duckdb")
        result = result.limit(limit)

    assert offset is not None, "offset is None"

    if not isinstance(offset, int):
        skip = offset
        skip = sg.select(skip).from_(table).subquery().sql(dialect="duckdb")
    elif not offset:
        return result
    else:
        skip = offset

    return result.offset(skip)


@translate_rel.register
def _distinct(op: ops.Distinct, *, table, **_):
    return sg.select(STAR).distinct().from_(table)


@translate_rel.register(ops.DropNa)
def _dropna(op: ops.DropNa, *, table, how, subset, **_):
    if subset is None:
        subset = [
            sg.column(name, table=table.alias_or_name) for name in op.table.schema.names
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
        return table

    try:
        return table.where(predicate)
    except AttributeError:
        return sg.select(STAR).from_(table).where(predicate)


@translate_rel.register
def _fillna(op: ops.FillNa, *, table, replacements, **_):
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
    return sg.select(*exprs).from_(table)


@translate_rel.register
def _view(op: ops.View, *, child, name: str, **_):
    # TODO: find a way to do this without creating a temporary view
    backend = op.child.to_expr()._find_backend()
    backend._create_temp_view(table_name=name, source=sg.select(STAR).from_(child))
    return sg.table(name)


@translate_rel.register
def _sql_string_view(op: ops.SQLStringView, query: str, **_: Any):
    table = sg.table(op.name)
    return sg.select(STAR).from_(table).with_(table, as_=query, dialect="duckdb")
