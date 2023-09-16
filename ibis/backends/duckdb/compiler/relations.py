from __future__ import annotations

import functools
from functools import partial

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.duckdb.compiler.values import translate_val


@functools.singledispatch
def translate_rel(op: ops.TableNode, **_):
    """Translate a table node into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_rel.register(ops.DummyTable)
def _dummy(op: ops.DummyTable, **kw):
    return sg.select(*map(partial(translate_val, **kw), op.values), dialect="duckdb")


@translate_rel.register(ops.PhysicalTable)
def _physical_table(op: ops.PhysicalTable, **_):
    return sg.expressions.Table(this=sg.to_identifier(op.name, quoted=True))


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
    sel = sg.select(*selections, dialect="duckdb").from_(
        from_ if from_ is not None else table, dialect="duckdb"
    )

    if join is not None:
        sel = sel.join(join)

    if predicates := op.predicates:
        if join is not None:
            sel = sg.select("*").from_(sel.subquery(kw["aliases"][op.table]))
        res = functools.reduce(
            lambda left, right: left.and_(right),
            (
                sg.condition(tr_val(predicate), dialect="duckdb")
                for predicate in predicates
            ),
        )
        sel = sel.where(res, dialect="duckdb")

    if sort_keys := op.sort_keys:
        sel = sel.order_by(*map(tr_val, sort_keys), dialect="duckdb")

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
        sel = sel.group_by(*map(tr_val_no_alias, group_keys), dialect="duckdb")

    if predicates := op.predicates:
        sel = sel.where(*map(tr_val_no_alias, predicates), dialect="duckdb")

    if having := op.having:
        sel = sel.having(*map(tr_val_no_alias, having), dialect="duckdb")

    if sort_keys := op.sort_keys:
        sel = sel.order_by(*map(tr_val_no_alias, sort_keys), dialect="duckdb")

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
                sg.condition(translate_val(predicate, **kw), dialect="duckdb")
                for predicate in predicates
            ),
        )
    else:
        on = None
    join_type = _JOIN_TYPES[type(op)]
    try:
        return left.join(right, join_type=join_type, on=on, dialect="duckdb")
    except AttributeError:
        select_args = [f"{left.alias_or_name}.*"]

        # select from both the left and right side of the join if the join
        # is not a filtering join (semi join or anti join); filtering joins
        # only return the left side columns
        if not isinstance(op, (ops.LeftSemiJoin, ops.LeftAntiJoin)):
            select_args.append(f"{right.alias_or_name}.*")
        return (
            sg.select(*select_args, dialect="duckdb")
            .from_(left, dialect="duckdb")
            .join(right, join_type=join_type, on=on, dialect="duckdb")
        )


@translate_rel.register
def _self_ref(op: ops.SelfReference, *, table, aliases, **kw):
    if (name := aliases.get(op)) is None:
        return table
    return sg.alias(table, name)


@translate_rel.register
def _query(op: ops.SQLQueryResult, *, aliases, **_):
    res = sg.parse_one(op.query, read="duckdb")
    return res.subquery(aliases.get(op, "_"))


_SET_OP_FUNC = {
    ops.Union: sg.union,
    ops.Intersection: sg.intersect,
    ops.Difference: sg.except_,
}


@translate_rel.register
def _set_op(op: ops.SetOp, *, left, right, **_):
    dialect = "duckdb"

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
def _limit(op: ops.Limit, *, table, n, offset, **kw):
    result = sg.select("*").from_(table)

    if isinstance(n, int):
        result = result.limit(n)
    elif n is not None:
        limit = translate_val(n, **kw)
        # TODO: calling `.sql` is a workaround for sqlglot not supporting
        # scalar subqueries in limits
        limit = sg.select(limit).from_(table).subquery().sql(dialect="duckdb")
        result = result.limit(limit)

    assert offset is not None, "offset is None"

    if not isinstance(offset, int):
        skip = translate_val(offset, **kw)
        skip = sg.select(skip).from_(table).subquery().sql(dialect="duckdb")
    elif not offset:
        return result
    else:
        skip = offset

    return result.offset(skip)


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
        return table.where(predicate, dialect="duckdb")
    except AttributeError:
        return sg.select("*").from_(table).where(predicate, dialect="duckdb")
