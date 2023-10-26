from __future__ import annotations

import functools

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import STAR


@functools.singledispatch
def translate_rel(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_rel.register(ops.DummyTable)
def dummy_table(op, *, values, **_):
    return sg.select(*values)


@translate_rel.register
def _physical_table(op: ops.PhysicalTable, **_):
    return sg.table(op.name)


@translate_rel.register(ops.DatabaseTable)
def table(op, *, name, namespace, **_):
    return sg.table(name, db=namespace.schema, catalog=namespace.database)


@translate_rel.register(ops.SelfReference)
def _self_ref(op, *, table, **_):
    return sg.alias(table, op.name)


_JOIN_TYPES = {
    ops.InnerJoin: "inner",
    ops.LeftJoin: "left",
    ops.RightJoin: "right",
    ops.OuterJoin: "full",
    ops.LeftAntiJoin: "left anti",
    ops.LeftSemiJoin: "left semi",
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

    alias = table.alias_or_name
    selections = tuple(
        replace_tables_with_star_selection(
            node,
            # replace the table name with the alias if the table is **not** a
            # join, because we may be selecting from a subquery or an aliased
            # table; otherwise we'll select from the _unaliased_ table or the
            # _child_ table, which may have a different alias than the one we
            # generated for the input table
            alias if from_ is None and join is None else None,
        )
        for node in selections
    ) or (STAR,)

    sel = sg.select(*selections).from_(from_ if from_ is not None else table)

    if join is not None:
        sel = sel.join(join)

    if predicates:
        if join is not None:
            sel = sg.select(STAR).from_(sel.subquery(alias))
        sel = sel.where(*predicates)

    if sort_keys:
        sel = sel.order_by(*sort_keys)

    return sel


@translate_rel.register
def _limit(op: ops.Limit, *, table, n, offset, **_):
    result = sg.select(STAR).from_(table)

    if n is not None:
        if not isinstance(n, int):
            limit = sg.select(n).from_(table).subquery()
        else:
            limit = n
        result = result.limit(limit)

    if not isinstance(offset, int):
        return result.offset(
            sg.select(offset).from_(table).subquery().sql("clickhouse")
        )

    return result.offset(offset) if offset != 0 else result


@translate_rel.register(ops.Aggregation)
def _aggregation(
    op: ops.Aggregation, *, table, metrics, by, having, predicates, sort_keys, **_
):
    selections = (by + metrics) or (STAR,)
    sel = sg.select(*selections).from_(table)

    if by:
        sel = sel.group_by(
            *(key.this if isinstance(key, sg.exp.Alias) else key for key in by)
        )

    if predicates:
        sel = sel.where(*predicates)

    if having:
        sel = sel.having(*having)

    if sort_keys:
        sel = sel.order_by(*sort_keys)

    return sel


_SET_OP_FUNC = {
    ops.Union: sg.union,
    ops.Intersection: sg.intersect,
    ops.Difference: sg.except_,
}


@translate_rel.register
def _set_op(op: ops.SetOp, *, left, right, distinct: bool = False, **_):
    if isinstance(left, sg.exp.Table):
        left = sg.select(STAR).from_(left)

    if isinstance(right, sg.exp.Table):
        right = sg.select(STAR).from_(right)

    func = _SET_OP_FUNC[type(op)]

    left = left.args.get("this", left)
    right = right.args.get("this", right)

    return func(left, right, distinct=distinct)


@translate_rel.register
def _distinct(op: ops.Distinct, *, table, **_):
    return sg.select(STAR).distinct().from_(table)
