from __future__ import annotations

import ast
from typing import Callable


def matches(value: ast.AST, pattern: ast.AST) -> bool:
    """Check whether `value` matches `pattern`."""
    # types must match exactly
    if type(value) != type(pattern):
        return False

    # primitive value, such as None, True, False etc
    if not isinstance(value, ast.AST) and not isinstance(pattern, ast.AST):
        return value == pattern

    fields = [
        (field, getattr(pattern, field))
        for field in pattern._fields
        if hasattr(pattern, field)
    ]
    return all(
        matches(getattr(value, field_name), field_value)
        for field_name, field_value in fields
    )


class Rewriter:
    """AST pattern matcher to enable rewrite rules."""

    def __init__(self):
        self.funcs: list[tuple[ast.AST, Callable[[ast.expr], ast.expr]]] = []

    def register(self, pattern):
        def wrapper(f):
            self.funcs.append((pattern, f))
            return f

        return wrapper

    def __call__(self, node):
        # TODO: more efficient way of doing this?
        for pattern, func in self.funcs:
            if matches(node, pattern):
                return func(node)
        return node


rewrite = Rewriter()
