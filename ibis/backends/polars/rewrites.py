from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.schema import Schema


@public
class PandasRename(ops.Relation):
    parent: ops.Relation
    mapping: FrozenDict[str, str]

    @classmethod
    def from_prefix(cls, parent, prefix):
        mapping = {k: f"{prefix}_{k}" for k in parent.schema}
        return cls(parent, mapping)

    @attribute
    def values(self):
        return FrozenDict(
            {to: ops.Field(self.parent, from_) for from_, to in self.mapping.items()}
        )

    @attribute
    def schema(self):
        return Schema(
            {self.mapping[name]: dtype for name, dtype in self.parent.schema.items()}
        )


@public
class PandasJoin(ops.Relation):
    left: ops.Relation
    right: ops.Relation
    left_on: VarTuple[ops.Value]
    right_on: VarTuple[ops.Value]
    how: str

    @attribute
    def values(self):
        return FrozenDict({**self.left.values, **self.right.values})

    @attribute
    def schema(self):
        return self.left.schema | self.right.schema


@public
class PandasAsofJoin(PandasJoin):
    left_by: VarTuple[ops.Value]
    right_by: VarTuple[ops.Value]
    operator: type


def split_join_predicates(left, right, predicates, only_equality=True):
    left_on = []
    right_on = []
    for pred in predicates:
        if left not in pred.relations or right not in pred.relations:
            # not a usual join predicate, so apply a trick by placing the
            # predicate to the left side and adding a literal True to the right
            # which the left side must be equal to
            left_on.append(pred)
            right_on.append(ops.Literal(True, dtype=dt.boolean))
        elif isinstance(pred, ops.Binary):
            if only_equality and not isinstance(pred, ops.Equals):
                raise TypeError("Only equality join predicates supported with pandas")
            if left in pred.left.relations and right in pred.right.relations:
                left_on.append(pred.left)
                right_on.append(pred.right)
            elif left in pred.right.relations and right in pred.left.relations:
                left_on.append(pred.right)
                right_on.append(pred.left)
            else:
                raise ValueError("Join predicate does not reference both tables")
        else:
            raise TypeError(f"Unsupported join predicate {pred}")

    return left_on, right_on


@replace(ops.JoinChain)
def rewrite_join(_, **kwargs):
    # TODO(kszucs): JoinTable.index can be used as a prefix
    prefixes = {}
    prefixes[_.first] = prefix = str(len(prefixes))
    left = PandasRename.from_prefix(_.first, prefix)

    for link in _.rest:
        prefixes[link.table] = prefix = str(len(prefixes))
        right = PandasRename.from_prefix(link.table, prefix)

        subs = {v: ops.Field(left, k) for k, v in left.values.items()}
        subs.update({v: ops.Field(right, k) for k, v in right.values.items()})
        preds = [pred.replace(subs, filter=ops.Value) for pred in link.predicates]

        # separate ASOF from the rest of the joins
        if link.how == "asof":
            on, *by = preds
            left_on, right_on = split_join_predicates(
                left, right, [on], only_equality=False
            )
            left_by, right_by = split_join_predicates(left, right, by)
            left = PandasAsofJoin(
                how="asof",
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
                left_by=left_by,
                right_by=right_by,
                operator=type(on),
            )
        else:
            # need to replace the fields in the predicates
            left_on, right_on = split_join_predicates(left, right, preds)
            left = PandasJoin(
                how=link.how,
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
            )

    subs = {v: ops.Field(left, k) for k, v in left.values.items()}
    fields = {k: v.replace(subs, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(left, fields)


@replace(ops.UnboundTable)
def bind_unbound_table(_, backend, **kwargs):
    return ops.DatabaseTable(name=_.name, schema=_.schema, source=backend)
