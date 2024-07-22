from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from public import public

import ibis.expr.operations as ops
from ibis import util
from ibis.common.deferred import Deferred
from ibis.common.egraph import DisjointSet
from ibis.common.exceptions import (
    ExpressionError,
    IbisInputError,
    IbisTypeError,
    InputTypeError,
    IntegrityError,
)
from ibis.expr.rewrites import flatten_predicates, peel_join_field
from ibis.expr.types.generic import Value
from ibis.expr.types.relations import (
    DerefMap,
    Table,
    bind,
    unwrap_aliases,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ibis.expr.operations.relations import JoinKind


def disambiguate_fields(
    how,
    predicates,
    equalities,
    left_fields,
    right_fields,
    left_template,
    right_template,
):
    """Resolve name collisions between the left and right tables."""
    collisions = set()
    left_template = left_template or "{name}"
    right_template = right_template or "{name}"

    if how == "inner" and util.all_of(predicates, ops.Equals):
        # for inner joins composed exclusively of equality predicates, we can
        # avoid renaming columns with colliding names if their values are
        # guaranteed to be equal due to the predicate
        equalities = equalities.copy()
        for pred in predicates:
            if isinstance(pred.left, ops.Field) and isinstance(pred.right, ops.Field):
                # disjoint sets are used to track the equality groups
                equalities.add(pred.left)
                equalities.add(pred.right)
                equalities.union(pred.left, pred.right)

    if how in ("semi", "anti"):
        # discard the right fields per left semi and left anty join semantics
        return left_fields, collisions, equalities

    fields = {}
    for name, field in left_fields.items():
        if name in right_fields:
            # there is an overlap between this field and a field from the right
            try:
                # check if the fields are equal due to equality predicates
                are_equal = equalities.connected(field, right_fields[name])
            except KeyError:
                are_equal = False
            if not are_equal:
                # there is a name collision and the fields are not equal, so
                # rename the field from the left according to the provided
                # template (which is the name itself by default)
                name = left_template.format(name=name)

        fields[name] = field

    for name, field in right_fields.items():
        if name in left_fields:
            # there is an overlap between this field and a field from the left
            try:
                # check if the fields are equal due to equality predicates
                are_equal = equalities.connected(field, left_fields[name])
            except KeyError:
                are_equal = False

            if are_equal:
                # even though there is a name collision, the fields are equal
                # due to equality predicates, so we can safely discard the
                # field from the right
                continue
            else:
                # there is a name collision and the fields are not equal, so
                # rename the field from the right according to the provided
                # template
                name = right_template.format(name=name)

        if name in fields:
            # we can still have collisions after multiple joins, or a wrongly
            # chosen template, so we need to track the collisions
            collisions.add(name)
        else:
            # the field name does not collide with any field from the left
            # and not occupied by any field from the right, so add it to the
            # fields mapping
            fields[name] = field

    return fields, collisions, equalities


def prepare_predicates(
    chain: ops.JoinChain,
    right: ops.Relation,
    predicates: Sequence[Any],
    comparison: type[ops.Comparison] = ops.Equals,
):
    """Bind and dereference predicates to the left and right tables.

    The responsibility of this function is twofold:
    1. Convert the various input values to valid predicates, including binding.
    2. Dereference the predicates one of the ops.JoinTable(s) in the join chain
       or the new JoinTable wrapping the right table. JoinTable(s) are used to
       ensure that all join participants are unique, even if the same table is
       joined multiple times.

    Since join predicates can be ambiguous sometimes, we do the two steps above
    in the same time so that we have more contextual information to resolve
    ambiguities.

    Possible inputs for the predicates:
    1. A python boolean literal, which is converted to a literal expression
    2. A boolean `Value` expression, which gets flattened and dereferenced.
       If there are comparison expressions where both sides depend on the same
       relation, then the left side is dereferenced to one of the join tables
       already part of the join chain, while the right side is dereferenced to
       the new join table wrapping the right table.
    3. A `Deferred` expression, which gets resolved on the left table and then
       the same path is followed as for `Value` expressions.
    4. A pair of expression-like objects, which are getting bound to the left
       and right tables respectively using the robust `bind` function handling
       several cases, including `Deferred` expressions, `Selector`s, literals,
       etc. Then the left are dereferenced to the join chain whereas the right
       to the new join table wrapping the right table.

    Parameters
    ----------
    chain
        The join chain
    right
        The right table
    predicates
        Predicates to bind and dereference, see the possible values above
    comparison
        The comparison operation to construct if the input is a pair of
        expression-like objects
    """
    reverse = {
        ops.Field(chain, k): v
        for k, v in chain.values.items()
        if isinstance(v, ops.Field)
    }
    deref_right = DerefMap.from_targets(right)
    deref_left = DerefMap.from_targets(chain.tables, extra=reverse)
    deref_both = DerefMap.from_targets([*chain.tables, right], extra=reverse)

    left, right = chain.to_expr(), right.to_expr()
    for pred in util.promote_list(predicates):
        if isinstance(pred, (Value, Deferred, bool)):
            for bound in bind(left, pred):
                yield deref_both.dereference(bound.op())
        else:
            if isinstance(pred, tuple):
                if len(pred) != 2:
                    raise ExpressionError("Join key tuple must be length 2")
                lk, rk = pred
            else:
                lk = rk = pred

            for lhs, rhs in zip(bind(left, lk), bind(right, rk)):
                lhs = deref_left.dereference(lhs.op())
                rhs = deref_right.dereference(rhs.op())
                yield comparison(lhs, rhs)


def finished(method):
    """Decorator to ensure the join chain is finished before calling a method."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self._finish(), *args, **kwargs)

    return wrapper


@public
class Join(Table):
    __slots__ = ("_collisions", "_equalities")

    def __init__(self, arg, collisions=(), equalities=()):
        assert isinstance(arg, ops.Node)
        if not isinstance(arg, ops.JoinChain):
            # coerce the input node to a join chain operation
            arg = ops.JoinReference(arg, identifier=0)
            arg = ops.JoinChain(arg, rest=(), values=arg.fields)
        super().__init__(arg)
        # the collisions and equalities are used to track the name collisions
        # and the equality groups join fields based on equality predicates;
        # these must be tracked in the join expression because the join chain
        # operation doesn't hold any information about `lname` and `rname`
        # parameters passed to the join methods and used to disambiguate field
        # names; the collisions are used to raise an error if there are any
        # name collisions after the join chain is finished
        object.__setattr__(self, "_collisions", collisions or set())
        object.__setattr__(self, "_equalities", equalities or DisjointSet())

    def _finish(self) -> Table:
        """Construct a valid table expression from this join expression."""
        if self._collisions:
            raise IntegrityError(f"Name collisions: {self._collisions}")
        return Table(self.op())

    @functools.wraps(Table.join)
    def join(
        self,
        right,
        predicates: Any = (),
        how: JoinKind = "inner",
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        if not isinstance(right, Table):
            raise IbisTypeError(
                f"Right side of join must be an Ibis table, got {type(right)}."
            )

        if how == "left_semi":
            how = "semi"
        elif how == "asof":
            raise IbisInputError("use table.asof_join(...) instead")

        chain = self.op()
        right = right.op()
        if not isinstance(right, ops.Reference):
            right = ops.JoinReference(right, identifier=chain.length)

        # bind and dereference the predicates
        preds = prepare_predicates(chain, right, predicates)
        preds = flatten_predicates(preds)
        if not preds and how not in {"cross", "positional"}:
            # if there are no predicates, default to every row matching unless
            # the join is a cross join, because a cross join already has this
            # behavior
            preds.append(ops.Literal(True, dtype="bool"))

        # calculate the fields based in lname and rname, this should be a best
        # effort to avoid collisions, but does not raise if there are any
        # if no disambiaution happens using a final .select() call, then
        # the finish() method will raise due to the name collisions
        values, collisions, equalities = disambiguate_fields(
            how=how,
            predicates=preds,
            equalities=self._equalities,
            left_fields=chain.values,
            right_fields=right.fields,
            left_template=lname,
            right_template=rname,
        )

        # construct a new join link and add it to the join chain
        link = ops.JoinLink(how, table=right, predicates=preds)
        chain = chain.copy(rest=chain.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(chain, collisions=collisions, equalities=equalities)

    @functools.wraps(Table.asof_join)
    def asof_join(
        self: Table,
        right: Table,
        on,
        predicates=(),
        tolerance=None,
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        predicates = util.promote_list(predicates)
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
            values = {**filtered.op().values, **self.op().values}

            return result.select(**values)

        chain = self.op()
        right = right.op()
        if not isinstance(right, ops.Reference):
            right = ops.JoinReference(right, identifier=chain.length)

        # TODO(kszucs): add extra validation for `on` with clear error messages
        (on,) = prepare_predicates(chain, right, [on], comparison=ops.GreaterEqual)
        preds = prepare_predicates(chain, right, predicates, comparison=ops.Equals)
        preds = [on, *preds]

        values, collisions, equalities = disambiguate_fields(
            how="asof",
            predicates=preds,
            equalities=self._equalities,
            left_fields=chain.values,
            right_fields=right.fields,
            left_template=lname,
            right_template=rname,
        )

        # construct a new join link and add it to the join chain
        link = ops.JoinLink("asof", table=right, predicates=preds)
        chain = chain.copy(rest=chain.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(chain, collisions=collisions, equalities=equalities)

    @functools.wraps(Table.cross_join)
    def cross_join(
        self: Table,
        right: Table,
        *rest: Table,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        left = self.join(right, how="cross", predicates=(), lname=lname, rname=rname)
        for table in rest:
            left = left.join(
                table, how="cross", predicates=(), lname=lname, rname=rname
            )
        return left

    @functools.wraps(Table.select)
    def select(self, *args, **kwargs):
        chain = self.op()
        values = self.bind(*args, **kwargs)
        values = unwrap_aliases(values)

        links = [link.table for link in chain.rest if link.how not in ("semi", "anti")]
        derefmap = DerefMap.from_targets([chain.first, *links])

        # if there are values referencing fields from the join chain constructed
        # so far, we need to replace them the fields from one of the join links
        values = {
            k: v.replace(peel_join_field, filter=ops.Value) for k, v in values.items()
        }
        values = {k: derefmap.dereference(v) for k, v in values.items()}

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
    fill_null = finished(Table.fill_null)
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
