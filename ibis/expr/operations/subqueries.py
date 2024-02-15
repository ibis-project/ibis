from __future__ import annotations

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.exceptions import IntegrityError
from ibis.expr.operations.core import Value
from ibis.expr.operations.relations import Relation  # noqa: TCH001


@public
class Subquery(Value):
    rel: Relation

    @attribute
    def relations(self):
        return frozenset()


@public
class ExistsSubquery(Subquery):
    dtype = dt.boolean
    shape = ds.columnar


@public
class ScalarSubquery(Subquery):
    shape = ds.scalar

    def __init__(self, rel):
        if len(rel.schema) != 1:
            raise IntegrityError(
                "Relation passed to ScalarSubquery() must have exactly one "
                f"column, got {len(rel.schema)}"
            )
        super().__init__(rel=rel)

    @attribute
    def value(self):
        (value,) = self.rel.values.values()
        return value

    @attribute
    def dtype(self):
        return self.value.dtype


@public
class InSubquery(Subquery):
    needle: Value

    dtype = dt.boolean
    shape = rlz.shape_like("needle")

    def __init__(self, rel, needle):
        if len(rel.schema) != 1:
            raise IntegrityError(
                "Relation passed to InSubquery() must have exactly one "
                f"column, got {len(rel.schema)}"
            )
        (value,) = rel.values.values()
        if not rlz.comparable(value, needle):
            raise IntegrityError(f"{needle!r} is not comparable to {value!r}")
        super().__init__(rel=rel, needle=needle)

    @attribute
    def value(self):
        (value,) = self.rel.values.values()
        return value

    @attribute
    def relations(self):
        return self.needle.relations
