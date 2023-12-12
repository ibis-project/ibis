from ibis.expr.types.relations import (
    bind,
    dereference_values,
    unwrap_aliases,
)
from public import public
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr, ValueExpr
from typing import Any
from collections.abc import Iterator, Mapping
from ibis.common.deferred import Deferred
from ibis.expr.analysis import flatten_predicates
from ibis.common.exceptions import ExpressionError, IntegrityError
from ibis import util
import ibis


def disambiguate_fields(how, left_fields, right_fields, lname, rname):
    if how in ("semi", "anti"):
        # discard the right fields
        return left_fields

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
        if name not in fields:
            fields[name] = field

    return fields


def prepare_predicates(left, right, predicates, deref_left, deref_right, deref_both):
    """Bind predicates to the left and right tables."""

    for pred in util.promote_list(predicates):
        if pred is True or pred is False:
            yield ops.Literal(pred, dtype="bool")
        elif isinstance(pred, ValueExpr):
            node = pred.op()
            yield node.replace(deref_both, filter=ops.Value)
        elif isinstance(pred, Deferred):
            # resolve deferred expressions on the left table
            node = pred.resolve(left).op()
            yield node.replace(deref_both, filter=ops.Value)
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
            left_value = left_value.op().replace(deref_left, filter=ops.Value)
            right_value = right_value.op().replace(deref_right, filter=ops.Value)

            yield ops.Equals(left_value, right_value).to_expr()


@public
class JoinExpr(TableExpr):
    def join(
        self,
        right,
        predicates: Any,
        # TODO(kszucs): add typehint about the possible join kinds
        how: str = "inner",
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        """Join with another table."""
        import pyarrow as pa
        import pandas as pd

        if isinstance(right, (pd.DataFrame, pa.Table)):
            right = ibis.memtable(right)
        elif not isinstance(right, TableExpr):
            raise TypeError(
                f"right operand must be a TableExpr, got {type(right).__name__}"
            )

        left = self.op()
        right = right.op()
        if isinstance(right, ops.SelfReference):
            deref_right = {}
        else:
            right = ops.SelfReference(right)
            deref_right = {v: ops.Field(right, k) for k, v in right.fields.items()}

        # construct the various dereference mappings
        deref_left = {ops.Field(left, k): v for k, v in left.fields.items()}
        deref_both = deref_left.copy()
        for table in self._relations():
            deref_both.update({v: ops.Field(table, k) for k, v in table.fields.items()})
        # finally add the right dereferences to take precedence
        deref_both.update(deref_right)

        # bind and dereference the predicates
        preds = prepare_predicates(
            left.to_expr(),
            right.to_expr(),
            predicates,
            deref_left=deref_left,
            deref_right=deref_right,
            deref_both=deref_both,
        )
        preds = flatten_predicates(list(preds))

        # calculate the fields based in lname and rname, this should be a best
        left_fields = left.values
        right_fields = {k: ops.Field(right, k) for k in right.schema}
        values = disambiguate_fields(how, left_fields, right_fields, lname, rname)

        # construct a new join link and add it to the join chain
        link = ops.JoinLink(how, table=right, predicates=preds)
        left = left.copy(rest=left.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(left)

    def select(self, *args, **kwargs):
        """Select expressions."""
        chain = self.op()
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)

        # if there are values referencing fields from the join chain constructed
        # so far, we need to replace them the fields from one of the join links
        subs = {ops.Field(chain, k): v for k, v in chain.fields.items()}
        tables = set(self._relations())
        for table in self._relations():
            for k, v in table.fields.items():
                if isinstance(v, ops.Field) and v.rel not in tables:
                    subs[v] = ops.Field(table, k)
        values = {k: v.replace(subs, filter=ops.Value) for k, v in values.items()}

        return self.finish(values)

    # TODO(kszucs): figure out a solution to automatically wrap all the
    # TableExpr methods including the docstrings and the signature
    def filter(self, *predicates):
        """Filter with `predicates`."""
        return self.finish().filter(*predicates)

    def order_by(self, *keys):
        """Order the join by the given keys."""
        return self.finish().order_by(*keys)

    def _relations(self) -> Iterator[ops.Relation]:
        """Yield all tables in this join expression."""
        node = self.op()
        yield node.first
        for join in node.rest:
            if join.how not in ("semi", "anti"):
                yield join.table

    def finish(self, values: Mapping[str, ops.Field] | None = None) -> TableExpr:
        """Construct a valid table expression from this join expression."""
        node = self.op()
        if values is None:
            # TODO(kszucs): clean this up with a nicer error message
            # raise if there are missing fields from either of the tables
            # raise on collisions
            collisions = []
            values = frozenset(self.op().values.values())
            for rel in self._relations():
                for k in rel.schema:
                    f = ops.Field(rel, k)
                    if f not in values:
                        collisions.append(f)
            if collisions:
                raise IntegrityError(f"Name collisions: {collisions}")
        else:
            node = node.copy(values=values)
        # important to explicitly create a TableExpr because .to_expr() would construct
        # a JoinExpr which should be finished again causing an infinite recursion
        return TableExpr(node)
