from __future__ import annotations

from ibis._tstring import Interpolation as Interpolation  # noqa: PLC0414
from ibis._tstring import PInterpolation as PInterpolation  # noqa: PLC0414
from ibis._tstring import PTemplate as PTemplate  # noqa: PLC0414
from ibis._tstring import Template as Template  # noqa: PLC0414
from ibis._tstring import t as _t


def t(template_string: str, /) -> Template:
    """Emulates a PEP 750 t-string literal for Python < 3.14.

    This function parses a string with f-string-like syntax and returns
    a `Template` object, correctly evaluating expressions in the caller's
    scope.

    Args:
        template_string: The string to parse, e.g., "Hello {name!r}".

    Returns:
        A `Template` instance containing the parsed static strings and
        evaluated interpolations.

    Example:
        >>> temp, unit = 22.43, "C"
        >>> template = t("Temperature: {temp:.1f} degrees {unit!s}")
        >>> template.strings
        ('Temperature: ', ' degrees ', '')
        >>> len(template.interpolations)
        2
        >>> template.interpolations[0]
        Interpolation(value=22.43, expression='temp', conversion=None, format_spec='.1f')
        >>> template.interpolations[1]
        Interpolation(value='C', expression='unit', conversion='s', format_spec='')
    """
    return _t(template_string, frame=1)


# @runtime_checkable
# class PInterpolation(Protocol):
#     """Protocol for an object that can be interpreted as an Interpolation."""

#     @property
#     def value(self) -> object: ...
#     @property
#     def expression(self) -> str: ...
#     @property
#     def conversion(self) -> Literal["a", "r", "s"] | None: ...
#     @property
#     def format_spec(self) -> str: ...


# @runtime_checkable
# class PTemplateString(Protocol):
#     """Protocol for an object that can be interpreted as a TemplateString."""

#     @property
#     def strings(self) -> tuple[str, ...]: ...
#     @property
#     def values(self) -> tuple[PInterpolation, ...]: ...
