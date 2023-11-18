from __future__ import annotations

# need a parallel Expression and Operation class hierarchy to decompose ops.Selection
# into proper relational algebra operations
################################ OPERATIONS ################################
import itertools
from abc import abstractmethod
from typing import Annotated, Any

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.deferred import Deferred, Item, deferred, var
from ibis.common.exceptions import IntegrityError
from ibis.common.patterns import Check, In, InstanceOf, _, pattern, replace
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Alias, Column, Node, Scalar, SortKey, Value
from ibis.expr.schema import Schema
from ibis.expr.types import Expr, literal
from ibis.expr.types import Value as ValueExpr
from ibis.selectors import Selector
from ibis.util import Namespace

p = Namespace(pattern, module=__name__)
d = Namespace(deferred, module=__name__)


class Relation(Node, Coercible):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = itertools.count()

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def fields(self):
        # used for dereferencing fields/values to the lowest relation in the
        # chain this should return references to parent relation's fields
        ...

    @property
    @abstractmethod
    def schema(self):
        ...

    def to_expr(self):
        return TableExpr(self)


class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


class ForeignField(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


def _check_integrity(values, allowed_parents):
    foreign_field = p.Field(~In(allowed_parents))
    for value in values:
        if disallowed := value.match(foreign_field, filter=Value):
            raise IntegrityError(
                f"Cannot add {disallowed!r} to projection, they belong to another relation"
            )
        # egyebkent csak scalar lehet (e.g. scalar subquery or a value based on literals)


class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), {parent})
        # for v in values.values():
        #     if v.find(ForeignField):
        #         print("found foreign field")
        # TODO(kszucs): raise if values depending on foreign fields are not scalar shaped
        super().__init__(parent=parent, values=values)

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


# TODO(kszucs): Subquery(value, outer_relation)


class Join(Node):
    how: str
    table: Relation
    predicates: VarTuple[Value[dt.Boolean]]


class JoinProject(Relation):
    first: Relation
    rest: VarTuple[Join]
    fields: FrozenDict[str, Field]

    def __init__(self, first, rest, fields):
        allowed_parents = {first}
        for join in rest:
            allowed_parents.add(join.table)
            _check_integrity(join.predicates, allowed_parents)
        _check_integrity(fields.values(), allowed_parents)
        super().__init__(first=first, rest=rest, fields=fields)

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, {parent})
        super().__init__(parent=parent, keys=keys)

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        # TODO(kszucs): use toolz.unique(predicates) to remove duplicates
        _check_integrity(predicates, {parent})
        super().__init__(parent=parent, predicates=predicates)

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
    metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

    @attribute
    def fields(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        # schema is consisting both by and metrics, use .from_tuples() to disallow
        # duplicate names in the schema
        groups = {k: v.dtype for k, v in self.groups.items()}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


class UnboundTable(Relation):
    name: str
    schema: Schema
    fields = FrozenDict()

    # @attribute
    # def fields(self):
    #     return {k: Field(self, k) for k in self.schema}


# class Subquery(Relation):
#     rel: Relation

#     @property
#     def schema(self):
#         return self.rel.schema

#     @property
#     def fields(self):
#         return self.rel.fields


################################ TYPES ################################


class TableExpr(Expr):
    def schema(self):
        return self.op().schema

    def __getattr__(self, key):
        return next(bind(self, key))

    def select(self, *args, **kwargs):
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)
        values = dereference_values(self.op(), values)
        # TODO(kszucs): windowization of reductions should happen here
        # 1. if a reduction if originating from self it should be turned into a window function
        # 2. if a scalar value is originating from a foreign table it should be turned into a scalar subquery
        node = Project(self, values)
        return node.to_expr()

    def filter(self, *predicates):
        preds = bind(self, predicates)
        preds = unwrap_aliases(preds)
        preds = dereference_values(self.op(), preds)
        # TODO(kszucs): add predicate flattening
        node = Filter(self, preds.values())
        return node.to_expr()

    def order_by(self, *keys):
        keys = bind(self, keys)
        keys = unwrap_aliases(keys).values()
        node = Sort(self, keys)
        return node.to_expr()

    def join(self, right, predicates, how="inner"):
        # construct an empty join chain and wrap it with a JoinExpr
        expr = JoinExpr(JoinProject(self, (), {}))
        # add the first join to the join chain and return the result
        return expr.join(right, predicates, how)

    def aggregate(self, groups, metrics):
        groups = bind(self, groups)
        metrics = bind(self, metrics)
        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)
        node = Aggregate(self, groups, metrics)
        return node.to_expr()

    def optimize(self):
        node = self.op()
        # apply the rewrites
        node = node.replace(
            complete_reprojection
            | subsequent_projects
            | subsequent_filters
            | subsequent_sorts
        )
        # return with a new TableExpr wrapping the optimized node
        return node.to_expr()


class JoinExpr(Expr):
    def join(self, right, predicates, how="inner"):
        node = self.op()
        # TODO(kszucs): need to do the usual input preparation here, binding,
        # unwrap_aliases, dereference_values, but the latter requires the
        # `field` property to be not empty in the JoinProject node

        # construct a new join node
        join = Join(how, table=right, predicates=predicates)

        # add the join to the join chain
        node = node.copy(rest=node.rest + (join,))

        # return with a new JoinExpr wrapping the new join chain
        return JoinExpr(node)

    def select(self, *args, **kwargs):
        # do the fields projection here
        # TODO(kszucs): need to do smarter binding here since references may
        # point to any of the relations in the join chain
        table = self.op().first.to_expr()
        values = bind(table, (args, kwargs))
        values = unwrap_aliases(values)
        # values = dereference_values(self.op(), values)

        # TODO(kszucs): go over the values and pull out the fields only, until
        # that just raise if the value is a computed expression
        for value in values.values():
            if not isinstance(value, Field):
                raise TypeError("Only fields can be selected in a join")

        return self.finish(values)

    # TODO(kszucs): figure out a solution to automatically wrap all the
    # TableExpr methods including the docstrings and the signature
    def where(self, *predicates):
        return self.finish().where(*predicates)

    def order_by(self, *keys):
        return self.finish().order_by(*keys)

    def guess(self):
        # TODO(kszucs): more advanced heuristics can be applied here
        # trying to figure out the join intent here
        node = self.op()
        parents = [node.first]
        for join in node.rest:
            parents.append(join.table)

        fields = {}
        for parent in parents:
            fields.update({k: Field(parent, k) for k in parent.schema})

        return fields

    def finish(self, fields=None):
        if fields is None:
            fields = self.guess()
        return self.op().copy(fields=fields).to_expr()

    def __getattr__(self, name):
        return next(bind(self.finish(), name))


def table(name, schema):
    return UnboundTable(name, schema).to_expr()


# TODO(kszucs): cover it with tests
def bind(table: TableExpr, value: Any) -> ir.Value:
    if isinstance(value, ValueExpr):
        yield value
    elif isinstance(value, TableExpr):
        yield from bind(table, tuple(value.schema().keys()))
    elif isinstance(value, str):
        yield Field(table, value).to_expr()
    elif isinstance(value, Deferred):
        yield value.resolve(table)
    elif isinstance(value, Selector):
        yield from value.expand(table)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from bind(table, v)
    elif isinstance(value, dict):
        for k, v in value.items():
            for val in bind(table, v):
                yield val.name(k)
    elif callable(value):
        yield value(table)
    else:
        yield literal(value)


# TODO(kszucs): cover it with tests
def unwrap_aliases(values):
    result = {}
    for value in values:
        node = value.op()
        if isinstance(node, Alias):
            result[node.name] = node.arg
        else:
            result[node.name] = node
    return result


@replace(p.Field)
def lookup_peeled_field(_, parent, mapping):
    name = mapping.get(_)
    if name is not None:
        return Field(parent, name)
    else:
        return ForeignField(_.rel, _.name)


def dereference_mapping(parent):
    result = {Field(parent, k): k for k in parent.schema}
    for k, v in parent.fields.items():
        while isinstance(v, Field):
            result[v] = k
            v = v.rel.fields.get(v.name)
    return result


def dereference_values(parent, values):
    mapping = dereference_mapping(parent)
    context = {"parent": parent, "mapping": mapping}
    return {
        k: v.replace(lookup_peeled_field, context=context, filter=Value)
        for k, v in values.items()
    }


################################ REWRITES ################################

name = var("name")

y = var("y")
values = var("values")


@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule) for v in _.predicates)
    return Filter(y.parent, y.predicates + preds)


# TODO(kszucs): this may work if the sort keys are not overlapping, need to revisit
@replace(p.Sort(y @ p.Sort))
def subsequent_sorts(_, y):
    return Sort(y.parent, y.keys + _.keys)


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()


# subqueries:
# 1. reduction passed to .filter() should be turned into a subquery
# 2. reduction passed to .select() with a foreign table should be turned into a subquery
