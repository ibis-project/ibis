import ast

import toolz


class NameFinder:
    """Helper class to find the unique names in an AST.
    """

    __slots__ = ()

    def find(self, node):
        typename = type(node).__name__
        method = getattr(self, 'find_{}'.format(typename), None)
        if method is None:
            fields = getattr(node, '_fields', None)
            if fields is None:
                return
            for field in fields:
                value = getattr(node, field)
                for result in self.find(value):
                    yield result
        else:
            for result in method(node):
                yield result

    def find_Name(self, node):
        # TODO not sure if this is robust to scope changes
        yield node

    def find_list(self, node):
        return list(toolz.concat(map(self.find, node)))

    def find_Call(self, node):
        if not isinstance(node.func, ast.Name):
            fields = node._fields
        else:
            fields = [field for field in node._fields if field != 'func']
        return toolz.concat(map(
            self.find, (getattr(node, field) for field in fields)
        ))


def find_names(node):
    """Return the unique :class:`ast.Name` instances in an AST.

    Parameters
    ----------
    node : ast.AST

    Returns
    -------
    unique_names : List[ast.Name]

    Examples
    --------
    >>> import ast
    >>> node = ast.parse('a + b')
    >>> names = find_names(node)
    >>> names  # doctest: +ELLIPSIS
    [<_ast.Name object at 0x...>, <_ast.Name object at 0x...>]
    >>> names[0].id
    'a'
    >>> names[1].id
    'b'
    """
    return list(toolz.unique(
        filter(None, NameFinder().find(node)),
        key=lambda node: (node.id, type(node.ctx))
    ))
