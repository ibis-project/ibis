from __future__ import annotations

import functools
from public import public
from typing import Any, Optional
from collections.abc import Iterator, Mapping

import ibis
import ibis.expr.operations as ops

from ibis import util
from ibis.expr.types import Table, Value
from ibis.common.deferred import Deferred
from ibis.expr.analysis import flatten_predicates
from ibis.common.exceptions import ExpressionError, IntegrityError
from ibis.expr.types.relations import (
    bind,
    dereference_values,
    dereference_mapping,
    unwrap_aliases,
)
from ibis.expr.operations.relations import JoinKind
from ibis.expr.rewrites import peel_join_field


def disambiguate_fields(how, left_fields, right_fields, lname, rname):
    collisions = set()

    if how in ("semi", "anti"):
        # discard the right fields per left semi and left anty join semantics
        return left_fields, collisions

    lname = lname or "{name}"
    rname = rname or "{name}"
    overlap = left_fields.keys() & right_fields.keys()

    fields = {}
    for name, field in left_fields.items():
        if name in overlap:
            name = lname.format(name=name)
        fields[name] = field
    for name, field in right_fields.items():
        if name in overlap:
            name = rname.format(name=name)
        # only add if there is no collision
        if name in fields:
            collisions.add(name)
        else:
            fields[name] = field

    return fields, collisions


def dereference_mapping_left(chain):
    # construct the list of join table we wish to dereference fields to
    rels = [chain.first]
    for link in chain.rest:
        if link.how not in ("semi", "anti"):
            rels.append(link.table)

    # create the dereference mapping suitable to disambiguate field references
    # from earlier in the relation hierarchy to one of the join tables
    subs = dereference_mapping(rels)

    # also allow to dereference fields of the join chain itself
    for k, v in chain.values.items():
        subs[ops.Field(chain, k)] = v

    return subs


def dereference_mapping_right(right):
    # the right table is wrapped in a JoinTable the uniqueness of the underlying
    # table which requires the predicates to be dereferenced to the wrapped
    return {v: ops.Field(right, k) for k, v in right.values.items()}


def dereference_sides(left, right, deref_left, deref_right):
    left = left.replace(deref_left, filter=ops.Value)
    right = right.replace(deref_right, filter=ops.Value)
    return left, right


def dereference_comparison_op(pred, deref_left, deref_right):
    left, right = dereference_sides(pred.left, pred.right, deref_left, deref_right)
    return pred.copy(left=left, right=right)


def dereference_value(pred, deref_left, deref_right):
    deref_both = {**deref_left, **deref_right}
    if isinstance(pred, ops.Comparison) and pred.left.relations == pred.right.relations:
        return dereference_comparison_op(pred, deref_left, deref_right)
    else:
        return pred.replace(deref_both, filter=ops.Value)


def prepare_predicates(
    left, right, predicates, deref_left, deref_right, comparison=ops.Equals
):
    """Bind and dereference predicates to the left and right tables."""

    left, right = left.to_expr(), right.to_expr()
    for pred in util.promote_list(predicates):
        if pred is True or pred is False:
            yield ops.Literal(pred, dtype="bool")
        elif isinstance(pred, Value):
            for node in flatten_predicates(pred.op()):
                yield dereference_value(node, deref_left, deref_right)
        elif isinstance(pred, Deferred):
            # resolve deferred expressions on the left table
            pred = pred.resolve(left).op()
            for node in flatten_predicates(pred):
                yield dereference_value(node, deref_left, deref_right)
        else:
            if isinstance(pred, tuple):
                if len(pred) != 2:
                    raise ExpressionError("Join key tuple must be length 2")
                lk, rk = pred
            else:
                lk = rk = pred

            # bind the predicates to the join chain
            (left_value,) = bind(left, lk)
            (right_value,) = bind(right, rk)

            # dereference the left value to one of the relations in the join chain
            left_value, right_value = dereference_sides(
                left_value.op(), right_value.op(), deref_left, deref_right
            )
            yield comparison(left_value, right_value)


def finished(method):
    """Decorator to ensure the join chain is finished before calling a method."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self._finish(), *args, **kwargs)

    return wrapper


@public
class Join(Table):
    __slots__ = ("_collisions",)

    def __init__(self, arg, collisions=None):
        assert isinstance(arg, ops.Node)
        if not isinstance(arg, ops.JoinChain):
            # coerce the input node to a join chain operation by first wrapping
            # the input relation in a JoinTable so that we can join the same
            # table with itself multiple times and to enable optimization
            # passes later on
            arg = ops.JoinTable(arg, index=0)
            arg = ops.JoinChain(arg, rest=(), values=arg.fields)
        super().__init__(arg)
        object.__setattr__(self, "_collisions", collisions or set())

    def _finish(self) -> Table:
        """Construct a valid table expression from this join expression."""
        if self._collisions:
            raise IntegrityError(f"Name collisions: {self._collisions}")
        return Table(self.op())

    @functools.wraps(Table.join)
    def join(  # noqa: D102
        self,
        right,
        predicates: Any,
        how: JoinKind = "inner",
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        import pyarrow as pa
        import pandas as pd

        # TODO(kszucs): factor out to a helper function
        if isinstance(right, (pd.DataFrame, pa.Table)):
            right = ibis.memtable(right)
        elif not isinstance(right, Table):
            raise TypeError(
                f"right operand must be a Table, got {type(right).__name__}"
            )

        if how == "left_semi":
            how = "semi"
        elif how == "asof":
            raise IbisInputError("use table.asof_join(...) instead")

        left = self.op()
        right = ops.JoinTable(right, index=left.length)
        subs_left = dereference_mapping_left(left)
        subs_right = dereference_mapping_right(right)

        # bind and dereference the predicates
        preds = list(
            prepare_predicates(
                left,
                right,
                predicates,
                deref_left=subs_left,
                deref_right=subs_right,
            )
        )
        if not preds and how != "cross":
            # if there are no predicates, default to every row matching unless
            # the join is a cross join, because a cross join already has this
            # behavior
            preds.append(ops.Literal(True, dtype="bool"))

        # calculate the fields based in lname and rname, this should be a best
        # effort to avoid collisions, but does not raise if there are any
        # if no disambiaution happens using a final .select() call, then
        # the finish() method will raise due to the name collisions
        values, collisions = disambiguate_fields(
            how, left.values, right.fields, lname, rname
        )

        # construct a new join link and add it to the join chain
        link = ops.JoinLink(how, table=right, predicates=preds)
        left = left.copy(rest=left.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(left, collisions=collisions)

    @functools.wraps(Table.asof_join)
    def asof_join(  # noqa: D102
        self: Table,
        right: Table,
        on,
        predicates=(),
        by=(),
        tolerance=None,
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        predicates = util.promote_list(predicates) + util.promote_list(by)
        if tolerance is not None:
            # `tolerance` parameter is mimicking the pandas API, but we express
            # it at the expression level by a sequence of operations:
            # 1. perform the `asof` join with the `on` an `predicates` parameters
            #    where the `on` parameter is an inequality predicate
            # 2. filter the asof join result using the `tolerance` parameter and
            #    the `on` parameter
            # 3. perform a left join between the original left table and the
            #    filtered asof join result using the `on` parameter but this
            #    time as an equality predicate
            if isinstance(on, str):
                # self is always a JoinChain so reference one of the join tables
                left_on = self.op().values[on].to_expr()
                right_on = right[on]
                on = left_on >= right_on
            elif isinstance(on, Value):
                node = on.op()
                if not isinstance(node, ops.Binary):
                    raise InputTypeError("`on` must be a comparison expression")
                left_on = node.left.to_expr()
                right_on = node.right.to_expr()
            else:
                raise TypeError("`on` must be a string or a ValueExpr")

            joined = self.asof_join(
                right, on=on, predicates=predicates, lname=lname, rname=rname
            )
            filtered = joined.filter(
                left_on <= right_on + tolerance, left_on >= right_on - tolerance
            )
            right_on = right_on.op().replace({right.op(): filtered.op()}).to_expr()

            # without joining twice the table would not contain the rows from
            # the left table that do not match any row from the right table
            # given the tolerance filter
            result = self.left_join(
                filtered, predicates=[left_on == right_on] + predicates
            )
            values = {**self.op().values, **filtered.op().values}
            return result.select(values)

        left = self.op()
        right = ops.JoinTable(right, index=left.length)
        subs_left = dereference_mapping_left(left)
        subs_right = dereference_mapping_right(right)

        # TODO(kszucs): add extra validation for `on` with clear error messages
        (on,) = prepare_predicates(
            left,
            right,
            [on],
            deref_left=subs_left,
            deref_right=subs_right,
            comparison=ops.GreaterEqual,
        )
        predicates = prepare_predicates(
            left,
            right,
            predicates,
            deref_left=subs_left,
            deref_right=subs_right,
            comparison=ops.Equals,
        )
        preds = [on, *predicates]

        values, collisions = disambiguate_fields(
            "asof", left.values, right.fields, lname, rname
        )

        # construct a new join link and add it to the join chain
        link = ops.JoinLink("asof", table=right, predicates=preds)
        left = left.copy(rest=left.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(left, collisions=collisions)

    @functools.wraps(Table.cross_join)
    def cross_join(  # noqa: D102
        self: Table,
        right: Table,
        *rest: Table,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        left = self.join(right, how="cross", predicates=(), lname=lname, rname=rname)
        for right in rest:
            left = left.join(
                right, how="cross", predicates=(), lname=lname, rname=rname
            )
        return left

    @functools.wraps(Table.select)
    def select(self, *args, **kwargs):  # noqa: D102
        chain = self.op()
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)

        # if there are values referencing fields from the join chain constructed
        # so far, we need to replace them the fields from one of the join links
        subs = dereference_mapping_left(chain)
        values = {
            k: v.replace(peel_join_field, filter=ops.Value) for k, v in values.items()
        }
        values = {k: v.replace(subs, filter=ops.Value) for k, v in values.items()}

        node = chain.copy(values=values)
        return Table(node)

    aggregate = finished(Table.aggregate)
    alias = finished(Table.alias)
    cast = finished(Table.cast)
    compile = finished(Table.compile)
    count = finished(Table.count)
    difference = finished(Table.difference)
    distinct = finished(Table.distinct)
    drop = finished(Table.drop)
    dropna = finished(Table.dropna)
    execute = finished(Table.execute)
    fillna = finished(Table.fillna)
    filter = finished(Table.filter)
    group_by = finished(Table.group_by)
    intersect = finished(Table.intersect)
    limit = finished(Table.limit)
    mutate = finished(Table.mutate)
    nunique = finished(Table.nunique)
    order_by = finished(Table.order_by)
    sample = finished(Table.sample)
    sql = finished(Table.sql)
    unbind = finished(Table.unbind)
    union = finished(Table.union)
    view = finished(Table.view)


public(JoinExpr=Join)
