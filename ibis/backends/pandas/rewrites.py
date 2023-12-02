from __future__ import annotations

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import In, replace
from ibis.common.typing import VarTuple
from ibis.expr.schema import Schema
from ibis.util import gen_name


# TODO(kszucs): possibly not needed
@public
class ColumnRef(ops.Value):
    name: str
    dtype: dt.DataType
    shape = ds.columnar


class PandasRelation(ops.Relation):
    pass


@public
class PandasRename(PandasRelation):
    parent: ops.Relation
    mapping: FrozenDict[str, str]

    @classmethod
    def from_prefix(cls, parent, prefix):
        mapping = {k: f"{prefix}_{k}" for k in parent.schema}
        return cls(parent, mapping)

    @attribute
    def fields(self):
        return FrozenDict(
            {to: ops.Field(self.parent, from_) for from_, to in self.mapping.items()}
        )

    @attribute
    def schema(self):
        return Schema(
            {self.mapping[name]: dtype for name, dtype in self.parent.schema.items()}
        )


@public
class PandasJoin(PandasRelation):
    left: ops.Relation
    right: ops.Relation
    left_on: VarTuple[ops.Value]
    right_on: VarTuple[ops.Value]
    how: str

    @attribute
    def fields(self):
        return FrozenDict({**self.left.fields, **self.right.fields})

    @attribute
    def schema(self):
        return self.left.schema | self.right.schema


@public
class PandasReduce(PandasRelation):
    parent: ops.Relation
    metrics: FrozenDict[str, ops.Scalar]

    @attribute
    def fields(self):
        return self.metrics

    @attribute
    def schema(self):
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples(metrics.items())


@public
class PandasAggregate(PandasRelation):
    parent: ops.Relation
    groups: VarTuple[str]
    metrics: FrozenDict[str, ops.Reduction]

    @attribute
    def fields(self):
        groups = {k: ops.Field(self.parent, k) for k in self.groups}
        return FrozenDict({**groups, **self.metrics})

    @attribute
    def schema(self):
        groups = {k: self.parent.schema[k] for k in self.groups}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


@replace(ops.Aggregate)
def aggregate_to_groupby(_):
    # add all the computed groups to the pre-projection
    select_derefs = {ops.Field(_.parent, k): k for k in _.parent.schema}
    for _k, v in _.groups.items():
        if v not in select_derefs:
            select_derefs[v] = gen_name("agg")

    # add all the computed dependencies of any reduction to the pre-projection
    # add all the reductions to the actual pandas aggregation
    reduction_derefs = {}
    for _k, v in _.metrics.items():
        for reduction in v.find_topmost(ops.Reduction):
            for arg in reduction.__args__:
                if (
                    isinstance(arg, ops.Value)
                    and arg.shape.is_columnar()
                    and arg not in select_derefs
                ):
                    select_derefs[arg] = gen_name("agg")
            if reduction not in reduction_derefs:
                reduction_derefs[reduction] = gen_name("agg")

    # STEP 1: construct the pre-projection
    proj = ops.Project(_.parent, {v: k for k, v in select_derefs.items()})

    # STEP 2: construct the pandas aggregation
    subs = {node: ColumnRef(name, node.dtype) for name, node in proj.fields.items()}
    groups = [select_derefs[node] for node in _.groups.values()]
    metrics = {name: node.replace(subs) for node, name in reduction_derefs.items()}
    if groups:
        agg = PandasAggregate(proj, groups, metrics)
    else:
        agg = PandasReduce(proj, metrics)

    # STEP 3: construct the post-projection
    subs = {node: ops.Field(agg, name) for node, name in reduction_derefs.items()}
    group_values = {
        name: ops.Field(agg, select_derefs[node]) for name, node in _.groups.items()
    }
    metric_values = {name: node.replace(subs) for name, node in _.metrics.items()}
    return ops.Project(agg, {**group_values, **metric_values})


def split_predicates(left, right, predicates):
    left_on = []
    right_on = []
    for pred in predicates:
        if left not in pred.relations or right not in pred.relations:
            # not a usual join predicate, so apply a trick by placing the
            # predicate to the left side and adding a literal True to the right
            # which the left side must be equal to
            left_on.append(pred)
            right_on.append(ops.Literal(True, dtype=dt.boolean))
        elif isinstance(pred, ops.Equals):
            if left in pred.left.relations and right in pred.right.relations:
                left_on.append(pred.left)
                right_on.append(pred.right)
            elif left in pred.right.relations and right in pred.left.relations:
                left_on.append(pred.right)
                right_on.append(pred.left)
            else:
                raise ValueError("Join predicate does not reference both tables")
        else:
            raise TypeError("Only equality join predicates supported with pandas")

    return left_on, right_on


@replace(ops.JoinChain)
def join_chain_to_nested_joins(_):
    prefixes = {}
    prefixes[_.first] = prefix = str(len(prefixes))
    left = PandasRename.from_prefix(_.first, prefix)

    for link in _.rest:
        prefixes[link.table] = prefix = str(len(prefixes))
        right = PandasRename.from_prefix(link.table, prefix)

        subs = {v: ops.Field(left, k) for k, v in left.fields.items()}
        subs.update({v: ops.Field(right, k) for k, v in right.fields.items()})
        preds = [pred.replace(subs, filter=ops.Value) for pred in link.predicates]

        # need to replace the fields in the predicates
        left_on, right_on = split_predicates(left, right, preds)

        left = PandasJoin(
            how=link.how,
            left=left,
            right=right,
            left_on=left_on,
            right_on=right_on,
        )

    subs = {v: ops.Field(left, k) for k, v in left.fields.items()}
    fields = {k: v.replace(subs, filter=ops.Value) for k, v in _.fields.items()}
    return ops.Project(left, fields)
