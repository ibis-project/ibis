from __future__ import annotations

import ast

import toolz


class NameFinder:
    """Helper class to find the unique names in an AST."""

    __slots__ = ()

    def find(self, node):
        typename = type(node).__name__
        method = getattr(self, f"find_{typename}", None)
        if method is None:
            fields = getattr(node, "_fields", None)
            if fields is None:
                return
            for field in fields:
                value = getattr(node, field)
                yield from self.find(value)
        else:
            yield from method(node)

    def find_Name(self, node):
        # TODO not sure if this is robust to scope changes
        yield node

    def find_list(self, node):
        return list(toolz.concat(map(self.find, node)))

    def find_Call(self, node):
        if not isinstance(node.func, ast.Name):
            fields = node._fields
        else:
            fields = [field for field in node._fields if field != "func"]
        return toolz.concat(map(self.find, (getattr(node, field) for field in fields)))


def find_names(node: ast.AST) -> list[ast.Name]:
    """Return the unique `ast.Name` instances in an AST.

    Examples
    --------
    >>> import ast
    >>> node = ast.parse("a + b")
    >>> names = find_names(node)
    >>> names
    [<....Name object at 0x...>, <....Name object at 0x...>]
    >>> names[0].id
    'a'
    >>> names[1].id
    'b'
    """
    return list(
        toolz.unique(
            filter(None, NameFinder().find(node)),
            key=lambda node: (node.id, type(node.ctx)),
        )
    )
