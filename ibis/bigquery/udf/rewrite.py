import ast


def matches(value, pattern):
    """Check whether `value` matches `pattern`.

    Parameters
    ----------
    value : ast.AST
    pattern : ast.AST

    Returns
    -------
    matched : bool
    """
    # types must match exactly
    if type(value) != type(pattern):
        return False

    # primitive value, such as None, True, False etc
    if not isinstance(value, ast.AST) and not isinstance(pattern, ast.AST):
        return value == pattern

    fields = [
        (field, getattr(pattern, field))
        for field in pattern._fields if hasattr(pattern, field)
    ]
    for field_name, field_value in fields:
        if not matches(getattr(value, field_name), field_value):
            return False
    return True


class Rewriter:
    """AST pattern matching to enable rewrite rules.

    Attributes
    ----------
    funcs : List[Tuple[ast.AST, Callable[ast.expr, [ast.expr]]]]
    """
    def __init__(self):
        self.funcs = []

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
