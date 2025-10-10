"""Operations for template strings (t-strings)."""

from __future__ import annotations

from itertools import zip_longest
from typing import TYPE_CHECKING, Optional, Protocol

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.optimizer.annotate_types import annotate_types
from typing_extensions import runtime_checkable

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Value

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ibis.backends.sql.datatypes import SqlglotType
    from ibis.expr.operations.relations import Relation
    from ibis.expr.types.generic import Value as ExprValue


@runtime_checkable
class IntoInterpolation(Protocol):
    """Protocol for an object that can be interpreted as a PEP 750 t-string Interpolation."""

    value: ExprValue
    expression: str


@runtime_checkable
class IntoTemplate(Protocol):
    """Protocol for an object that can be interpreted as a PEP 750 t-string Template."""

    strings: tuple[str, ...]
    interpolations: tuple[IntoInterpolation, ...]


@public
class TemplateSQL(Value):
    strings: VarTuple[str]
    values: VarTuple[Value]
    dialect: Optional[str] = None
    """The SQL dialect the template was written in.

    eg if t'CAST({val} AS REAL)', you should use 'sqlite',
    since REAL is a sqlite-specific concept.
    """

    def __init__(self, strings, values, dialect: str | None = None):
        super().__init__(strings=strings, values=values, dialect=dialect or "duckdb")
        if self.dtype.is_unknown():
            raise TypeError(
                f"Could not infer the dtype of the template expression with sql:\n{self.sql_for_inference}"
            )

    @classmethod
    def from_template(
        cls, template: IntoTemplate, /, *, dialect: str | None = None
    ) -> TemplateSQL:
        return cls(
            strings=template.strings,
            values=[interp.value for interp in template.interpolations],
            dialect=dialect,
        )

    @attribute
    def shape(self):
        if not self.values:
            return ds.scalar
        return rlz.highest_precedence_shape(self.values)

    @attribute
    def dtype(self) -> dt.DataType:
        parsed = sg.parse_one(self.sql_for_inference, dialect=self.dialect)
        annotated = annotate_types(parsed, dialect=self.dialect)
        sqlglot_type = annotated.type
        return self.type_mapper.to_ibis(sqlglot_type)

    @attribute
    def relations(self) -> frozenset[Relation]:
        children = (n.relations for n in self.values)
        return frozenset().union(*children)

    @property
    def sql_for_inference(self) -> str:
        parts: list[str] = []
        for part in self.parts:
            if isinstance(part, str):
                parts.append(part)
            else:
                ibis_type: dt.DataType = part.dtype
                null_sqlglot_value = sge.cast(
                    sge.null(), self.type_mapper.from_ibis(ibis_type)
                )
                parts.append(null_sqlglot_value.sql(self.dialect))
        return "".join(parts)

    @property
    def type_mapper(self) -> SqlglotType:
        return get_type_mapper(self.dialect)

    @property
    def parts(self):
        def iter() -> Iterator[str | Value]:
            for s, i in zip_longest(self.strings, self.values):
                if s:
                    yield s
                if i:
                    yield i

        return tuple(iter())


def get_type_mapper(dialect: str | None) -> SqlglotType:
    """Get the type mapper for the given SQL dialect."""
    import importlib

    module = importlib.import_module(f"ibis.backends.sql.compilers.{dialect}")
    compiler_instance = module.compiler
    return compiler_instance.type_mapper
