from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.patterns import Check, Object, replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import _, p
from ibis.expr.schema import Schema


@public
class Select(ops.Relation):
    parent: ops.Relation
    joins: VarTuple[ops.JoinLink] = ()
    selections: FrozenDict[str, ops.Value] = {}
    predicates: VarTuple[ops.Value[dt.Boolean]] = ()
    sort_keys: VarTuple[ops.SortKey] = ()

    @attribute
    def fields(self):
        return self.selections

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.selections.items()})


@replace(p.Project)
def project_to_select(_):
    return Select(_.parent, selections=_.values)


@replace(p.Filter)
def filter_to_select(_):
    return Select(_.parent, selections=_.fields, predicates=_.predicates)


@replace(p.Sort)
def sort_to_select(_):
    return Select(_.parent, selections=_.fields, sort_keys=_.keys)


@replace(p.JoinChain)
def join_chain_to_select(_):
    return Select(_.first, selections=_.fields, joins=_.rest)


@replace(
    Object(Select, Object(Select))
    & ~Check(_.parent.find(ops.WindowFunction, filter=ops.Value))
)
def merge_select_select(_):
    subs = {ops.Field(_.parent, k): v for k, v in _.parent.fields.items()}
    joins = tuple(j.replace(subs) for j in _.joins)
    selections = {k: v.replace(subs) for k, v in _.selections.items()}
    predicates = tuple(p.replace(subs) for p in _.predicates)
    sort_keys = tuple(s.replace(subs) for s in _.sort_keys)

    return Select(
        _.parent.parent,
        joins=_.parent.joins + joins,
        selections=selections,
        predicates=_.parent.predicates + predicates,
        sort_keys=_.parent.sort_keys + sort_keys,
    )


def sqlize(node):
    rules = join_chain_to_select | project_to_select | filter_to_select | sort_to_select
    step1 = node.replace(rules)
    step2 = step1.replace(merge_select_select)
    return step2
