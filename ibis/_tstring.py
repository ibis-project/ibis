"""A Backport of PEP 750 Template Strings (t-strings)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from itertools import zip_longest
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "Interpolation",
    "Template",
    "t",
]

# Regex to find and parse an f-string-like interpolation.
# It captures:
# 1. The main expression.
# 2. An optional debug specifier (=).
# 3. An optional conversion specifier (!r, !s, or !a).
# 4. An optional format specifier (:...).
_INTERPOLATION_RE = re.compile(
    r"""
    \{
        # The core expression, non-greedy
        (?P<expression>.+?)
        # Optional debug specifier
        (?P<debug>=)?
        # Optional conversion, one of !r, !s, or !a
        (?P<conversion>![rsa])?
        # Optional format spec, starting with a colon, non-greedy until }
        (?P<format_spec>:[^}]*)?
    }
    """,
    re.VERBOSE | re.DOTALL,
)

if sys.version_info >= (3, 10):
    dataclass_extra_args = {"slots": True}
else:
    dataclass_extra_args = {}


@runtime_checkable
class PInterpolation(Protocol):
    """Protocol for an object that can be interpreted as a PEP 750 t-string Interpolation."""

    @property
    def value(self) -> object: ...
    @property
    def expression(self) -> str: ...
    @property
    def conversion(self) -> Literal["a", "r", "s"] | None: ...
    @property
    def format_spec(self) -> str: ...


@dataclass(frozen=True, eq=False, **dataclass_extra_args)
class Interpolation:
    """Emulates the string.templatelib.Interpolation class from PEP 750.

    Represents an expression inside a template string.
    """

    value: object
    expression: str
    conversion: Literal["a", "r", "s"] | None = None
    format_spec: str = ""

    def __eq__(self, value: object) -> bool:
        """Template and Interpolation instances compare with object identity (is)."""
        return self is value

    def __hash__(self) -> int:
        """Hash based on identity."""
        return id(self)


@runtime_checkable
class PTemplate(Protocol):
    """Protocol for an object that can be interpreted as a PEP 750 t-string Interpolation."""

    @property
    def strings(self) -> tuple[str, ...]: ...
    @property
    def interpolations(self) -> tuple[PInterpolation, ...]: ...


@dataclass(frozen=True, eq=False, **dataclass_extra_args)
class Template:
    """Emulates the string.templatelib.Template class from PEP 750.

    Represents a parsed t-string literal.
    """

    strings: tuple[str, ...]
    """
    A non-empty tuple of the string parts of the template,
    with N+1 items, where N is the number of interpolations
    in the template.
    """
    interpolations: tuple[Interpolation, ...]
    """
    A tuple of the interpolation parts of the template.
    This will be an empty tuple if there are no interpolations.
    """

    @property
    def values(self) -> tuple[object, ...]:
        """A tuple of the `value` attributes of each Interpolation in the template.

        This will be an empty tuple if there are no interpolations.
        """
        return tuple(interp.value for interp in self.interpolations)

    def __iter__(self) -> Iterator[str | Interpolation]:
        """Iterate over the string parts and interpolations in the template.

        These may appear in any order. Empty strings will not be included.
        """
        for s, i in zip_longest(self.strings, self.interpolations):
            if s:
                yield s
            if i:
                yield i

    def __add__(self, other: Template) -> Template:
        """Adds two templates together."""
        # lazy duck-typing isinstance check
        if not hasattr(other, "strings") or not hasattr(other, "interpolations"):
            return NotImplemented
        *first, final = self.strings
        other_first, *other_rest = other.strings
        return self.__class__(
            strings=(*first, final + other_first, *other_rest),
            interpolations=self.interpolations + other.interpolations,
        )

    def __eq__(self, value: object) -> bool:
        """Template and Interpolation instances compare with object identity (is)."""
        return self is value

    def __hash__(self) -> int:
        """Hash based on identity."""
        return id(self)

    def __str__(self) -> NoReturn:
        """Explicitly disallowed."""
        raise TypeError("Template instances cannot be converted to strings directly.")


def t(template_string: str, /, frame: int | None = None) -> Template:
    # Get the execution frame of the caller to evaluate expressions in their scope.
    # sys._getframe(0) is the current frame
    # sys._getframe(1) is the frame of the caller
    # So if we called t("foo"), we want the frame where t("foo") was called, eg one above this
    # If we called t("foo", frame=1), we want 1 frame above where t("foo", frame=1) was called,
    # which is ACTUALLY frame=2 in this function
    if frame is None:
        frame = 1
    else:
        frame = frame + 1
    caller_frame = sys._getframe(frame)
    caller_globals = caller_frame.f_globals
    caller_locals = caller_frame.f_locals

    strings = []
    interpolations = []
    last_end = 0

    for match in _INTERPOLATION_RE.finditer(template_string):
        # Add the static string part before this interpolation
        strings.append(template_string[last_end : match.start()])
        last_end = match.end()

        groups = match.groupdict()

        # The debug specifier is syntactic sugar. It modifies both the
        # preceding string part and the interpolation itself.
        if groups["debug"]:
            # t'{value=}' becomes t'value={value!r}'
            # t'{value=:fmt}' becomes t'value={value!s:fmt}'

            # Find the position of the '=' in the original match string
            # so we can split the expression and the '=' (with whitespace)
            expr_with_possible_ws = groups["expression"]
            # Find the '=' at the end (possibly with whitespace before/after)
            eq_index = expr_with_possible_ws.rfind("=")
            if eq_index != -1:
                expr_for_static = expr_with_possible_ws[: eq_index + 1]
                # Remove trailing whitespace and the '=' for evaluation
                expr_for_eval = expr_with_possible_ws[:eq_index]
                # Strip all whitespace from both ends for evaluation
                expr_for_eval = expr_for_eval.strip()
                # Remove any trailing '=' if present (shouldn't be, but for safety)
                if expr_for_eval.endswith("="):
                    expr_for_eval = expr_for_eval[:-1].rstrip()
            else:
                expr_for_static = expr_with_possible_ws + "="
                expr_for_eval = expr_with_possible_ws.strip()

            # Prepend 'expression=' (with whitespace) to the *current* static string.
            strings[-1] += expr_for_static

            # For debug specifier, strip trailing '=' and whitespace for evaluation
            # (already done above)

            if groups["conversion"]:
                raise SyntaxError("f-string: cannot specify both conversion and '='")

            # If a format spec is present, conversion becomes 's'. Otherwise, 'r'.
            conv_char = "s" if groups["format_spec"] else "r"
            expression_to_eval = expr_for_eval
        else:
            conv_char = groups["conversion"][1] if groups["conversion"] else None
            expression_to_eval = groups["expression"]

        fmt_spec = groups["format_spec"][1:] if groups["format_spec"] else ""

        # Dedent multiline expressions for evaluation
        import textwrap

        expr_eval_str = textwrap.dedent(expression_to_eval)

        # Evaluate the expression to get its value using the caller's context
        try:
            value = eval(expr_eval_str, caller_globals, caller_locals)  # noqa: S307
        except Exception as e:
            # Re-raise with more context
            msg = f"Failed to evaluate expression '{expression_to_eval}': {e}"
            raise RuntimeError(msg) from e

        interpolations.append(
            Interpolation(
                value=value,
                expression=expression_to_eval,
                conversion=conv_char,
                format_spec=fmt_spec,
            )
        )

    # Add the final static string part after the last interpolation
    strings.append(template_string[last_end:])

    return Template(strings=tuple(strings), interpolations=tuple(interpolations))
