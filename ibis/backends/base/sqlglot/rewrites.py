from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import p
from ibis.expr.schema import Schema
from ibis.util import gen_name


@public
class Select(ops.Relation):
    parent: ops.Relation
    selections: FrozenDict[str, ops.Value]
    predicates: VarTuple[ops.Value[dt.Boolean]]
    sort_keys: VarTuple[ops.SortKey]

    @attribute
    def fields(self):
        return self.selections

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.selections.items()})


@replace(p.Project)
def project_to_select(_):
    return Select(_.parent, _.values, (), ())


@replace(p.Filter)
def filter_to_select(_):
    return Select(_.parent, _.parent.fields, _.predicates, ())


@replace(p.Sort)
def sort_to_select(_):
    return Select(_.parent, _.parent.fields, (), _.sort_keys)


@replace(p.Project(Select))
def merge_project_select(_):
    subs = {ops.Field(_.parent, k): v for k, v in _.parent.fields.items()}
    values = {k: v.replace(subs) for k, v in _.values.items()}
    return Select(_.parent.parent, values, _.parent.predicates, _.parent.sort_keys)


@replace(p.Filter(Select))
def merge_filter_select(_):
    subs = {ops.Field(_.parent, k): v for k, v in _.parent.fields.items()}
    predicates = tuple(p.replace(subs) for p in _.predicates)
    return Select(_.parent.parent, _.parent.selections, predicates, _.parent.sort_keys)


@replace(p.Sort(Select))
def merge_sort_select(_):
    subs = {ops.Field(_.parent, k): v for k, v in _.parent.fields.items()}
    keys = tuple(s.replace(subs) for s in _.keys)
    return Select(_.parent.parent, _.parent.selections, _.parent.predicates, keys)


def sqlize(node):
    rules = (
        merge_project_select
        | merge_filter_select
        | merge_sort_select
        | project_to_select
        | filter_to_select
        | sort_to_select
    )
    return node.replace(rules)
