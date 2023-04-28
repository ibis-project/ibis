from __future__ import annotations

import functools
from functools import partial

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.clickhouse.compiler.values import translate_val


@functools.singledispatch
def translate_rel(op: ops.TableNode, **_):
    """Translate a table node into sqlglot."""
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate_rel.register(ops.DummyTable)
def _dummy(op: ops.DummyTable, **kw):
    return sg.select(
        *map(partial(translate_val, **kw), op.values), dialect="clickhouse"
    )


@translate_rel.register(ops.PhysicalTable)
def _physical_table(op: ops.PhysicalTable, **_):
    return sg.parse_one(op.name, into=sg.exp.Table)


@translate_rel.register(ops.Selection)
def _selection(op: ops.Selection, *, table, needs_alias=False, **kw):
    # needs_alias should never be true here in explicitly, but it may get
    # passed via a (recursive) call to translate_val
    assert not needs_alias, "needs_alias is True"
    if needs_alias := isinstance(op.table, ops.Join) and not isinstance(
        op.table, (ops.LeftSemiJoin, ops.LeftAntiJoin)
    ):
        args = table.this.args
        from_ = args["from"]
        (join,) = args["joins"]
    else:
        from_ = join = None
    tr_val = partial(translate_val, needs_alias=needs_alias, **kw)
    selections = tuple(map(tr_val, op.selections)) or "*"
    sel = sg.select(*selections, dialect="clickhouse").from_(
        from_ if from_ is not None else table, dialect="clickhouse"
    )

    if join is not None:
        sel = sel.join(join)

    if predicates := op.predicates:
        if join is not None:
            sel = sg.select("*").from_(sel.subquery(kw["aliases"][op.table]))
        res = functools.reduce(
            lambda left, right: left.and_(right),
            (
                sg.condition(tr_val(predicate), dialect="clickhouse")
                for predicate in predicates
            ),
        )
        sel = sel.where(res, dialect="clickhouse")

    if sort_keys := op.sort_keys:
        sel = sel.order_by(*map(tr_val, sort_keys), dialect="clickhouse")

    return sel


@translate_rel.register(ops.Aggregation)
def _aggregation(op: ops.Aggregation, *, table, **kw):
    tr_val = partial(translate_val, **kw)
    tr_val_no_alias = partial(translate_val, render_aliases=False, **kw)

    by = tuple(map(tr_val, op.by))
    metrics = tuple(map(tr_val, op.metrics))
    selections = (by + metrics) or "*"
    sel = sg.select(*selections).from_(table)

    if group_keys := op.by:
        sel = sel.group_by(*map(tr_val_no_alias, group_keys), dialect="clickhouse")

    if predicates := op.predicates:
        sel = sel.where(*map(tr_val_no_alias, predicates), dialect="clickhouse")

    if having := op.having:
        sel = sel.having(*map(tr_val_no_alias, having), dialect="clickhouse")

    if sort_keys := op.sort_keys:
        sel = sel.order_by(*map(tr_val_no_alias, sort_keys), dialect="clickhouse")

    return sel


_JOIN_TYPES = {
    ops.InnerJoin: "INNER",
    ops.AnyInnerJoin: "ANY",
    ops.LeftJoin: "LEFT OUTER",
    ops.AnyLeftJoin: "LEFT ANY",
    ops.RightJoin: "RIGHT OUTER",
    ops.OuterJoin: "FULL OUTER",
    ops.CrossJoin: "CROSS",
    ops.LeftSemiJoin: "LEFT SEMI",
    ops.LeftAntiJoin: "LEFT ANTI",
    ops.AsOfJoin: "LEFT ASOF",
}


@translate_rel.register
def _join(op: ops.Join, *, left, right, **kw):
    predicates = op.predicates
    if predicates:
        on = functools.reduce(
            lambda left, right: left.and_(right),
            (
                sg.condition(translate_val(predicate, **kw), dialect="clickhouse")
                for predicate in predicates
            ),
        )
    else:
        on = None
    join_type = _JOIN_TYPES[type(op)]
    try:
        return left.join(right, join_type=join_type, on=on, dialect="clickhouse")
    except AttributeError:
        select_args = [f"{left.alias_or_name}.*"]

        # select from both the left and right side of the join if the join
        # is not a filtering join (semi join or anti join); filtering joins
        # only return the left side columns
        if not isinstance(op, (ops.LeftSemiJoin, ops.LeftAntiJoin)):
            select_args.append(f"{right.alias_or_name}.*")
        return (
            sg.select(*select_args, dialect="clickhouse")
            .from_(left, dialect="clickhouse")
            .join(right, join_type=join_type, on=on, dialect="clickhouse")
        )


@translate_rel.register
def _self_ref(op: ops.SelfReference, *, table, aliases, **kw):
    if (name := aliases.get(op)) is None:
        return table
    return sg.alias(table, name)


@translate_rel.register
def _query(op: ops.SQLQueryResult, *, aliases, **_):
    res = sg.parse_one(op.query, read="clickhouse")
    return res.subquery(aliases.get(op, "_"))


_SET_OP_FUNC = {
    ops.Union: sg.union,
    ops.Intersection: sg.intersect,
    ops.Difference: sg.except_,
}


@translate_rel.register
def _set_op(op: ops.SetOp, *, left, right, **_):
    dialect = "clickhouse"

    if isinstance(left, sg.exp.Table):
        left = sg.select("*", dialect=dialect).from_(left, dialect=dialect)

    if isinstance(right, sg.exp.Table):
        right = sg.select("*", dialect=dialect).from_(right, dialect=dialect)

    return _SET_OP_FUNC[type(op)](
        left.args.get("this", left),
        right.args.get("this", right),
        distinct=op.distinct,
        dialect=dialect,
    )


@translate_rel.register
def _limit(op: ops.Limit, *, table, **kw):
    n = op.n
    limited = sg.select("*").from_(table).limit(n)

    if offset := op.offset:
        limited = limited.offset(offset)
    return limited


@translate_rel.register
def _distinct(_: ops.Distinct, *, table, **kw):
    return sg.select("*").distinct().from_(table)


@translate_rel.register(ops.DropNa)
def _dropna(op: ops.DropNa, *, table, **kw):
    how = op.how

    if op.subset is None:
        columns = [ops.TableColumn(op.table, name) for name in op.table.schema.names]
    else:
        columns = op.subset

    if columns:
        raw_predicate = functools.reduce(
            ops.And if how == "any" else ops.Or,
            map(ops.NotNull, columns),
        )
    elif how == "all":
        raw_predicate = ops.Literal(False, dtype=dt.bool)
    else:
        raw_predicate = None

    if not raw_predicate:
        return table

    tr_val = partial(translate_val, **kw)
    predicate = tr_val(raw_predicate)
    try:
        return table.where(predicate, dialect="clickhouse")
    except AttributeError:
        return sg.select("*").from_(table).where(predicate, dialect="clickhouse")
