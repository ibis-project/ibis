from __future__ import annotations

import itertools
import typing
from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, TypeVar

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.exceptions import IbisTypeError, IntegrityError, RelationError
from ibis.common.grounds import Concrete
from ibis.common.patterns import Between, InstanceOf
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations.core import Alias, Column, Node, Scalar, Value
from ibis.expr.operations.sortkeys import SortKey
from ibis.expr.schema import Schema
from ibis.formats import TableProxy  # noqa: TCH001

T = TypeVar("T")

Unaliased = Annotated[T, ~InstanceOf(Alias)]
NonSortKey = Annotated[T, ~InstanceOf(SortKey)]


@public
class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        from ibis.expr.types import Table

        if isinstance(value, Relation):
            return value
        elif isinstance(value, Table):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def values(self) -> FrozenDict[str, Value]:
        """A mapping of column names to expressions which build up the relation.

        This attribute is heavily used in rewrites as well as during field
        dereferencing in the API layer. The returned expressions must only
        originate from parent relations, depending on the relation type.
        """

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """The schema of the relation.

        All relations must have a well-defined schema.
        """
        ...

    @property
    def fields(self) -> FrozenDict[str, Column]:
        """A mapping of column names to fields of the relation.

        This calculated property shouldn't be overridden in subclasses since it
        is mostly used for convenience.
        """
        return FrozenDict({k: Field(self, k) for k in self.schema})

    def to_expr(self):
        from ibis.expr.types import Table

        return Table(self)


@public
class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    def __init__(self, rel, name):
        if name not in rel.schema:
            columns_formatted = ", ".join(map(repr, rel.schema.names))
            raise IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )
        super().__init__(rel=rel, name=name)

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]

    @attribute
    def relations(self):
        return frozenset({self.rel})


def _check_integrity(values, allowed_parents):
    for value in values:
        for rel in value.relations:
            if rel not in allowed_parents:
                raise IntegrityError(
                    f"Cannot add {value!r} to projection, they belong to another relation"
                )


@public
class Project(Relation):
    parent: Relation
    values: FrozenDict[str, NonSortKey[Unaliased[Value]]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), {parent})
        super().__init__(parent=parent, values=values)

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


class Simple(Relation):
    parent: Relation

    @attribute
    def values(self):
        return self.parent.fields

    @attribute
    def schema(self):
        return self.parent.schema


# TODO(kszucs): remove in favor of View
@public
class SelfReference(Simple):
    _uid_counter = itertools.count()

    identifier: Optional[int] = None

    def __init__(self, parent, identifier):
        if identifier is None:
            identifier = next(self._uid_counter)
        super().__init__(parent=parent, identifier=identifier)

    @attribute
    def values(self):
        return FrozenDict()


JoinKind = Literal[
    "inner",
    "left",
    "right",
    "outer",
    "asof",
    "semi",
    "anti",
    "any_inner",
    "any_left",
    "cross",
]


@public
class JoinTable(Simple):
    index: int


@public
class JoinLink(Node):
    how: JoinKind
    table: JoinTable
    predicates: VarTuple[Value[dt.Boolean]]


@public
class JoinChain(Relation):
    first: JoinTable
    rest: VarTuple[JoinLink]
    values: FrozenDict[str, Unaliased[Value]]

    def __init__(self, first, rest, values):
        allowed_parents = {first}
        assert first.index == 0
        for join in rest:
            assert join.table.index == len(allowed_parents)
            allowed_parents.add(join.table)
            _check_integrity(join.predicates, allowed_parents)
        _check_integrity(values.values(), allowed_parents)
        super().__init__(first=first, rest=rest, values=values)

    @property
    def length(self):
        return len(self.rest) + 1

    @attribute
    def schema(self):
        return Schema({k: v.dtype.copy(nullable=True) for k, v in self.values.items()})

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.Join(self)


@public
class Sort(Simple):
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, {parent})
        super().__init__(parent=parent, keys=keys)


@public
class Filter(Simple):
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        from ibis.expr.rewrites import ReductionLike

        for pred in predicates:
            if pred.find(ReductionLike, filter=Value):
                raise IntegrityError(
                    f"Cannot add {pred!r} to filter, it is a reduction which "
                    "must be converted to a scalar subquery first"
                )
            if pred.relations and parent not in pred.relations:
                raise IntegrityError(
                    f"Cannot add {pred!r} to filter, they belong to another relation"
                )
        super().__init__(parent=parent, predicates=predicates)


@public
class Limit(Simple):
    # TODO(kszucs): dynamic limit should contain ScalarSubqueries rather than
    # plain scalar values
    n: typing.Union[int, Scalar[dt.Integer], None] = None
    offset: typing.Union[int, Scalar[dt.Integer]] = 0


@public
class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Unaliased[Column]]
    metrics: FrozenDict[str, Unaliased[Scalar]]

    def __init__(self, parent, groups, metrics):
        _check_integrity(groups.values(), {parent})
        _check_integrity(metrics.values(), {parent})
        if duplicates := groups.keys() & metrics.keys():
            raise RelationError(
                f"Cannot add {duplicates} to aggregate, they are already in the groupby"
            )
        super().__init__(parent=parent, groups=groups, metrics=metrics)

    @attribute
    def values(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class Set(Relation):
    left: Relation
    right: Relation
    distinct: bool = False
    values = FrozenDict()

    def __init__(self, left, right, **kwargs):
        # convert to dictionary first, to get key-unordered comparison semantics
        if dict(left.schema) != dict(right.schema):
            raise RelationError("Table schemas must be equal for set operations")
        elif left.schema.names != right.schema.names:
            # rewrite so that both sides have the columns in the same order making it
            # easier for the backends to implement set operations
            cols = {name: Field(right, name) for name in left.schema.names}
            right = Project(right, cols)
        super().__init__(left=left, right=right, **kwargs)

    @attribute
    def schema(self):
        return self.left.schema


@public
class Union(Set):
    pass


@public
class Intersection(Set):
    pass


@public
class Difference(Set):
    pass


@public
class PhysicalTable(Relation):
    name: str
    values = FrozenDict()


@public
class Namespace(Concrete):
    database: Optional[str] = None
    schema: Optional[str] = None


@public
class UnboundTable(PhysicalTable):
    schema: Schema
    namespace: Namespace = Namespace()


@public
class DatabaseTable(PhysicalTable):
    schema: Schema
    source: Any
    namespace: Namespace = Namespace()


@public
class InMemoryTable(PhysicalTable):
    schema: Schema
    data: TableProxy


@public
class SQLQueryResult(Relation):
    """A table sourced from the result set of a select query."""

    query: str
    schema: Schema
    source: Any
    values = FrozenDict()


@public
class View(PhysicalTable):
    """A view created from an expression."""

    # TODO(kszucs): rename it to parent
    child: Relation

    @attribute
    def schema(self):
        return self.child.schema


@public
class SQLStringView(Relation):
    """A view created from a SQL string."""

    child: Relation
    query: str
    schema: Schema
    values = FrozenDict()


@public
class DummyTable(Relation):
    values: FrozenDict[str, Value]

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class FillNa(Simple):
    """Fill null values in the table."""

    replacements: typing.Union[Value[dt.Numeric | dt.String], FrozenDict[str, Any]]


@public
class DropNa(Simple):
    """Drop null values in the table."""

    how: typing.Literal["any", "all"]
    subset: Optional[VarTuple[Column]] = None


@public
class Sample(Simple):
    """Sample performs random sampling of records in a table."""

    fraction: Annotated[float, Between(0, 1)]
    method: typing.Literal["row", "block"]
    seed: typing.Union[int, None] = None


@public
class Distinct(Simple):
    """Distinct is a table-level unique-ing operation."""


# TODO(kszucs): support t.select(*t) syntax by implementing Table.__iter__()
