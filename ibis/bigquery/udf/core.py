"""Translate a Python AST to JavaScript
"""

import ast

from collections import ChainMap
import contextlib
import functools
import inspect
import textwrap

import six

import ibis.expr.datatypes as dt

from ibis.bigquery.udf.find import find_names
from ibis.bigquery.udf.rewrite import rewrite


class SymbolTable(ChainMap):
    """ChainMap subclass implementing scope for the translator.

    Notes
    -----
    JavaScript requires declarations in strict mode, so to implement this we
    shove a "let" at the beginning of every variable name if it doesn't already
    exist in the current scope.
    """
    def __getitem__(self, key):
        if key not in self:
            self[key] = key
            return 'let {}'.format(key)
        return key


def indent(lines, spaces=4):
    """Indent `lines` by `spaces` spaces.

    Parameters
    ----------
    lines : Union[str, List[str]]
        A string or list of strings to indent
    spaces : int
        The number of spaces to indent `lines`

    Returns
    -------
    indented_lines : str
    """
    if isinstance(lines, six.string_types):
        text = [lines]
    text = '\n'.join(lines)
    return textwrap.indent(text, ' ' * spaces)


def semicolon(f):
    """Add a semicolon to the result of a visit_* call.

    Parameters
    ----------
    f : callable
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs) + ';'
    return wrapper


@rewrite.register(ast.Call(func=ast.Name(id='print')))
def rewrite_print(node):
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='console', ctx=ast.Load()),
            attr='log',
            ctx=ast.Load()
        ),
        args=node.args,
        keywords=node.keywords,
    )


@rewrite.register(ast.Call(func=ast.Name(id='len')))
def rewrite_len(node):
    assert len(node.args) == 1
    return ast.Attribute(value=node.args[0], attr='length', ctx=ast.Load())


@rewrite.register(ast.Call(func=ast.Attribute(attr='append')))
def rewrite_append(node):
    return ast.Call(
        func=ast.Attribute(
            value=node.func.value,
            attr='push',
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=node.keywords,
    )


@rewrite.register(
    ast.Call(func=ast.Attribute(value=ast.Name(id='Array'), attr='from_'))
)
def rewrite_array_from(node):
    return ast.Call(
        func=ast.Attribute(value=node.func.value, attr='from'),
        args=node.args,
        keywords=node.keywords,
    )


class PythonToJavaScriptTranslator:
    constructor_map = {
        'list': 'Array',
        'Array': 'Array',
        'Date': 'Date',

        'dict': 'Object',
        'Map': 'Map',
        'WeakMap': 'WeakMap',

        'str': 'String',
        'String': 'String',

        'set': 'Set',
        'Set': 'Set',
        'WeakSet': 'WeakSet',
    }

    def __init__(self, function):
        self.function = function
        self.source = textwrap.dedent(inspect.getsource(function))
        self.ast = ast.parse(self.source)
        self.scope = SymbolTable()
        self.current_function = None
        self.current_class = None
        self.is_generator = False
        self.is_nested_definition = False

    def compile(self):
        return self.visit(self.ast)

    def visit(self, node):
        node = rewrite(node)
        typename = node.__class__.__name__
        method_name = 'visit_{}'.format(typename)
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(
                '{!r} nodes not yet implemented'.format(method_name)
            )
        assert callable(method)

        result = method(node)
        return result

    def visit_Name(self, node):
        if self.current_class is not None and node.id == 'self':
            return 'this'
        return node.id

    def visit_Yield(self, node):
        self.is_generator = True
        return 'yield {}'.format(self.visit(node.value))

    def visit_YieldFrom(self, node):
        self.is_generator = True
        return 'yield* {}'.format(self.visit(node.value))

    @semicolon
    def visit_Assign(self, node):
        try:
            target, = node.targets
        except ValueError:
            raise NotImplementedError(
                'Only single assignment supported for now'
            )

        if not isinstance(target, (ast.Name, ast.Subscript, ast.Attribute)):
            raise NotImplementedError(
                'Only index, attribute, and variable name assigment '
                'supported, got {}'.format(type(target).__name__)
            )

        is_name = isinstance(target, ast.Name)
        compiled_target = self.visit(target)
        if (not is_name or
                (self.current_class is not None and
                    compiled_target.startswith('this.'))):
            self.scope[compiled_target] = compiled_target
        return '{} = {}'.format(
            self.scope[compiled_target],
            self.visit(node.value)
        )

    def translate_special_method(self, name):
        return {
            '__init__': 'constructor'
        }.get(name, name)

    def visit_FunctionDef(self, node):
        self.current_function = node

        is_property_getter = any(
            getattr(dec, 'id', None) == 'property'
            for dec in node.decorator_list
        )

        if self.current_class is None:  # not a method
            if is_property_getter:
                raise TypeError(
                    'Functions cannot be properties, only methods can'
                )
            prefix = 'function'
        else:
            if is_property_getter and self.is_generator:
                raise TypeError('generator methods cannot be properties')
            prefix = 'get ' * is_property_getter

        with self.local_scope():
            body = indent(map(self.visit, node.body))

            if self.is_generator:
                prefix += '* '
            else:
                prefix += ' ' * (self.current_class is None)

            lines = [
                prefix +
                self.translate_special_method(node.name) +
                '({}) {{'.format(self.visit(node.args)),
                body,
                '}'
            ]

            self.current_function = None
            self.is_generator = False
        return '\n'.join(lines)

    @semicolon
    def visit_Return(self, node):
        return 'return {}'.format(self.visit(node.value))

    def visit_Add(self, node):
        return '+'

    def visit_Sub(self, node):
        return '-'

    def visit_Mult(self, node):
        return '*'

    def visit_Div(self, node):
        return '/'

    def visit_FloorDiv(self, node):
        assert False, 'should never reach FloorDiv'

    def visit_Pow(self, node):
        assert False, 'should never reach Pow'

    def visit_UnaryOp(self, node):
        return '({}{})'.format(self.visit(node.op), self.visit(node.operand))

    def visit_USub(self, node):
        return '-'

    def visit_UAdd(self, node):
        return '+'

    def visit_BinOp(self, node):
        left, op, right = node.left, node.op, node.right

        if isinstance(op, ast.Pow):
            return 'Math.pow({}, {})'.format(
                self.visit(left), self.visit(right)
            )
        elif isinstance(op, ast.FloorDiv):
            return 'Math.floor({} / {})'.format(
                self.visit(left), self.visit(right)
            )
        return '({} {} {})'.format(
            self.visit(left), self.visit(op), self.visit(right)
        )

    def visit_NameConstant(self, node):
        value = node.value
        if value is True:
            return 'true'
        elif value is False:
            return 'false'
        assert value is None, \
            'value is not True and is not False, must be None, got {}'.format(
                value
            )
        return 'null'

    def visit_Str(self, node):
        return repr(node.s)

    def visit_Num(self, node):
        return repr(node.n)

    def visit_List(self, node):
        return '[{}]'.format(', '.join(map(self.visit, node.elts)))

    def visit_Tuple(self, node):
        # tuples becomes lists in javascript
        return '[{}]'.format(', '.join(map(self.visit, node.elts)))

    def visit_Dict(self, node):
        return '{{{}}}'.format(
            ', '.join(
                '[{}]: {}'.format(self.visit(key), self.visit(value))
                for key, value in zip(node.keys, node.values)
            )
        )

    @semicolon
    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Starred(self, node):
        return '...{}'.format(self.visit(node.value))

    def visit_Call(self, node):
        thing_to_call = self.visit(node.func)
        constructors = self.__class__.constructor_map
        args = ', '.join(map(self.visit, node.args))
        try:
            thing_to_call = constructors[thing_to_call]
        except KeyError:
            format_string = '{}({})'
        else:
            format_string = '(new {}({}))'
        return format_string.format(thing_to_call, args)

    def visit_Attribute(self, node):
        return '{}.{}'.format(self.visit(node.value), node.attr)

    def visit_For(self, node):
        lines = [
            'for (let {} of {}) {{'.format(
                self.visit(node.target), self.visit(node.iter)
            )
        ]
        with self.local_scope():
            lines.append(indent(map(self.visit, node.body)))
        lines.append('}')
        return '\n'.join(lines)

    def visit_While(self, node):
        lines = ['while ({}) {{'.format(self.visit(node.test))]
        with self.local_scope():
            lines.append(indent(map(self.visit, node.body)))
        lines.append('}')
        return '\n'.join(lines)

    @semicolon
    def visit_Break(self, node):
        return 'break'

    @semicolon
    def visit_Continue(self, node):
        return 'continue'

    def visit_Eq(self, node):
        return '==='

    def visit_NotEq(self, node):
        return '!=='

    def visit_Or(self, node):
        return '||'

    def visit_And(self, node):
        return '&&'

    def visit_BoolOp(self, node):
        return '({})'.format(
            ' {} '.format(self.visit(node.op)).join(
                map(self.visit, node.values)
            )
        )

    def visit_Lt(self, node):
        return '<'

    def visit_LtE(self, node):
        return '<='

    def visit_Gt(self, node):
        return '>'

    def visit_GtE(self, node):
        return '>='

    def visit_Compare(self, node):
        rights = node.comparators
        ops = node.ops

        left = node.left
        comparisons = []
        for op, right in zip(ops, rights):
            comparisons.append(
                '({} {} {})'.format(
                    self.visit(left),
                    self.visit(op),
                    self.visit(right)
                )
            )
            left = right
        return ' && '.join(comparisons)

    @semicolon
    def visit_AugAssign(self, node):
        return '{} {}= {}'.format(
            self.visit(node.target),
            self.visit(node.op),
            self.visit(node.value),
        )

    def visit_Module(self, node):
        return '\n\n'.join(map(self.visit, node.body))

    def visit_arg(self, node):
        if self.current_class is not None and node.arg == 'self':
            return ''
        return node.arg

    def visit_arguments(self, node):
        args = list(filter(None, map(self.visit, node.args[:])))
        vararg = node.vararg
        if vararg is not None:
            args.append('...{}'.format(vararg.arg))
        return ', '.join(args)

    def visit_Lambda(self, node):
        args = node.args
        generated_args = self.visit(args)
        return '(({}) => {})'.format(generated_args, self.visit(node.body))

    @contextlib.contextmanager
    def local_scope(self):
        """Assign symbols to local variables.
        """
        self.scope = self.scope.new_child()
        try:
            yield self.scope
        finally:
            self.scope = self.scope.parents

    def visit_If(self, node):
        lines = ['if ({}) {{'.format(self.visit(node.test))]

        with self.local_scope():
            lines.append(indent(map(self.visit, node.body)))
            lines.append('}')

        if node.orelse:
            lines[-1] += ' else {'
            with self.local_scope():
                lines.append(indent(map(self.visit, node.orelse)))
                lines.append('}')
        return '\n'.join(lines)

    def visit_IfExp(self, node):
        return '({} ? {} : {})'.format(
            self.visit(node.test),
            self.visit(node.body),
            self.visit(node.orelse),
        )

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Subscript(self, node):
        return '{}[{}]'.format(self.visit(node.value), self.visit(node.slice))

    def visit_ClassDef(self, node):
        self.current_class = node
        bases = node.bases

        lines = ['class {}'.format(node.name)]
        if bases:
            lines[-1] += ' extends {}'.format(
                ', '.join(map(self.visit, bases))
            )
        lines[-1] += ' {'
        lines.append(indent(map(self.visit, node.body)))
        lines.append('}')
        self.current_class = None
        self.__class__.constructor_map[node.name] = node.name
        return '\n'.join(lines)

    def visit_Not(self, node):
        return '!'

    def visit_ListComp(self, node):
        """Generate a curried lambda function

        [x + y for x, y in [[1, 4], [2, 5], [3, 6]]]

        becomes

        [[1, 4], [2, 5], [3, 6]]].map(([x, y]) => x + y)
        """
        try:
            generator, = node.generators
        except ValueError:
            raise NotImplementedError(
                'Only single loop comprehensions are allowed'
            )

        names = find_names(generator.target)
        argslist = [ast.arg(arg=name.id, annotation=None) for name in names]
        if len(names) <= 1:
            signature = ast.arguments(
                args=argslist,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            )
        else:
            signature = ast.List(elts=argslist, ctx=ast.Load())

        array = generator.iter
        lam_sig = functools.partial(ast.Lambda, args=signature)

        filters = generator.ifs
        if filters:
            filt = ast.BoolOp(op=ast.And(), values=filters)
            # array.filter
            method = ast.Attribute(value=array, attr='filter', ctx=ast.Load())
            # array.filter(func)
            array = ast.Call(
                func=method, args=[lam_sig(body=filt)], keywords=[])

        method = ast.Attribute(value=array, attr='map', ctx=ast.Load())
        mapped = ast.Call(
            func=method, args=[lam_sig(body=node.elt)], keywords=[])
        result = self.visit(mapped)
        return result

    def visit_Delete(self, node):
        return '\n'.join(
            'delete {};'.format(self.visit(target)) for target in node.targets
        )


if __name__ == '__main__':
    from ibis.bigquery.api import udf

    @udf(
        input_type=[dt.double, dt.double, dt.int64],
        output_type=dt.Array(dt.double),
        strict=False
    )
    def my_func(a, b, n):
        class Rectangle:
            def __init__(self, width, height):
                self.width = width
                self.height = height

            @property
            def area(self):
                return self.width * self.height

            @property
            def perimeter(self):
                return self.width * 2 + self.height * 2

            def foobar(self, n):
                for i in range(n):
                    yield i

        def sum(values):
            result = 0
            for value in values:
                result += value
            console.log(result)  # noqa: F821
            return values.reduce(lambda a, b: a + b, 0)

        def range(n):
            i = 0
            while i < n:
                yield i
                i += 1

        some_stuff = [x + y for x, y in [[1, 4], [2, 5], [3, 6]] if 2 < x < 3]
        some_stuff1 = [range(x) for x in [1, 2, 3]]
        some_stuff2 = [x + y for x, y in [(1, 4), (2, 5), (3, 6)]]
        print(some_stuff)
        print(some_stuff1)
        print(some_stuff2)

        x = 1
        y = 2
        x = 3
        values = []
        for i in range(10):  # noqa: F821
            values.append(i)

        i = 0
        foo = 2
        bar = lambda x: x  # noqa: E731
        bazel = lambda x: y  # noqa: E731
        while i < n:
            foo = bar(bazel(10))
            i += 1
            console.log(i)  # noqa: F821

        foo = 2

        if i == 10 and (y < 2 or i != 42):
            y += 2
        else:
            y -= 2

        z = 42.0
        w = 3
        w = not False
        yyz = None
        print(yyz)  # noqa: F821
        foobar = x < y < z < w  # x < y and y < z
        foobar = 1
        baz = foobar // 3
        console.log(baz)  # noqa: F821

        my_obj = {'a': 1, 'b': 2}  # noqa: F841

        z = (x if y else b) + 2 + foobar
        foo = Rectangle(1, 2)
        nnn = len(values)
        return [sum(values) - a + b * y ** -x, z, foo.width, nnn]
    print(my_func.js)
