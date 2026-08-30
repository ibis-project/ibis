from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ibis._tstring import Interpolation, PInterpolation, PTemplate, Template
from ibis.common.bases import FrozenSlotted
from ibis.common.deferred import Resolver, resolver
from ibis.expr import datatypes as dt
from ibis.expr.operations.template import TemplateSQLValue, interleave

if TYPE_CHECKING:
    from ibis.expr import types as ir
    from ibis.expr.operations.template import Dialect


class TemplateValueResolver(FrozenSlotted, Resolver):
    __slots__ = ("dialect", "dtype", "template")
    template: PTemplate
    dialect: Dialect
    dtype: dt.DataType | None

    def __init__(
        self,
        template,
        dialect: Dialect | None = None,
        dtype: dt.IntoDtype | None = None,
    ):
        if dialect is None:
            dialect = "duckdb"
        dtype = dt.dtype(dtype) if dtype is not None else None
        super().__init__(template=template, dialect=dialect, dtype=dtype)

    def __repr__(self):
        vals = [i.value for i in self.template.interpolations]
        parts = interleave(self.template.strings, vals)
        repr_parts = [str(part) for part in parts]
        template = "".join(repr_parts)
        return (
            f"{type(self).__name__}("
            f"template={template!r}, "
            f"dialect={self.dialect!r}, "
            f"dtype={self.dtype!r})"
        )

    def resolve(self, context: dict[str, Any]) -> ir.Value:
        resolved_template = resolve_template_values(self.template, context)
        sql_value_op = TemplateSQLValue.from_template(
            resolved_template,
            dialect=self.dialect,
            dtype=self.dtype,
        )
        return sql_value_op.to_expr()


def resolve_template_values(template: PTemplate, context: dict[str, Any]) -> Template:
    """Take a PTemplate, and return a Template with of the interpolation values resolved."""

    def ensure_resolved(i: PInterpolation) -> Interpolation:
        resolver_obj = resolver(i.value)
        resolved = resolver_obj.resolve(context)
        return Interpolation(
            value=resolved,
            expression=i.expression,
            conversion=i.conversion,
            format_spec=i.format_spec,
        )

    resolved_interpolations = tuple(ensure_resolved(i) for i in template.interpolations)
    return Template(strings=template.strings, interpolations=resolved_interpolations)
