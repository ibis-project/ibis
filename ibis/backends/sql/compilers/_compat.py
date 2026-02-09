"""Compatibility layer for SQLGlot expressions."""

from __future__ import annotations

import sqlglot.expressions as sge


def _get_arg_name(expr_cls: type[sge.Expression], *args: str) -> str:
    if (arg := next(filter(expr_cls.arg_types.__contains__, args), None)) is not None:
        return arg
    raise ValueError(f"None of {args} are valid arguments for {expr_cls.__name__}")


# "precompute" common arg names
# sqglot >= 28.0 renamed arg types that mirror python keywords to include a trailing underscore
# to keep the code backward compatible, we take the first valid arg name
WITH_ARG = _get_arg_name(sge.Select, "with", "with_")
EXCEPT_ARG = _get_arg_name(sge.Star, "except", "except_")


def _supports_special_float_literals() -> bool:
    try:
        sge.Literal.number("binary_double_nan")
    except Exception:
        return False
    else:
        return True


_SPECIAL_FLOATS_AS_IDENTIFIER = not _supports_special_float_literals()


def special_float_literal(name: str) -> sge.Expression:
    if _SPECIAL_FLOATS_AS_IDENTIFIER:
        return sge.to_identifier(name)
    return sge.Literal.number(name)