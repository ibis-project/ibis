from __future__ import annotations

import builtins
import sys
import tempfile

import pytest

from ibis.backends.bigquery.udf.core import PythonToJavaScriptTranslator, SymbolTable


def test_symbol_table():
    symbols = SymbolTable()
    assert symbols["a"] == "let a"
    assert symbols["a"] == "a"


def compile(f):
    return PythonToJavaScriptTranslator(f).compile()


def test_function_def(snapshot):
    def f(a, b):
        return a + b

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_variable_declaration(snapshot):
    def f():
        c = 1
        return c + 2

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_yield(snapshot):
    def f(a):
        yield from [1, 2, 3]

    js = compile(f)
    snapshot.assert_match(js, "out.js")


@pytest.mark.skipif(sys.platform == "win32", reason="Skip on Windows")
def test_yield_from(snapshot):
    d = {}

    with tempfile.NamedTemporaryFile("r+") as f:
        f.write(
            """\
def f(a):
    yield from [1, 2, 3]"""
        )
        f.seek(0)
        code = builtins.compile(f.read(), f.name, "exec")
        exec(code, d)
        f = d["f"]
        js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_assign(snapshot):
    def f():
        a = 1
        a = 2
        print(a)  # noqa: T201
        return 1

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


@pytest.mark.parametrize("op", [add, sub, mul, div])
def test_binary_operators(op, snapshot):
    js = compile(op)
    snapshot.assert_match(js, "out.js")


def test_pow(snapshot):
    def f():
        a = 1
        return a**2

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_floordiv(snapshot):
    def f():
        a = 1
        return a // 2

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_unary_minus(snapshot):
    def f():
        a = 1
        return -a

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_unary_plus(snapshot):
    def f():
        a = 1
        return +a

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_true_false_none(snapshot):
    def f():
        a = True
        b = False
        c = None
        return a if c != None else b  # noqa: E711

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_str(snapshot):
    def f():
        a = "abc"
        return a

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_tuple(snapshot):
    def f():
        a = "a", "b", "c"
        return a

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_dict(snapshot):
    def f():
        a = {"a": 1, "b": 2}
        return a

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_continue(snapshot):
    def f():
        i = 0
        for i in [1, 2, 3]:
            if i == 1:
                continue
        return i

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_lambda_with_splat(snapshot):
    def f():
        def sum(sequence):
            total = 0
            for value in sequence:
                total += value
            return total

        splat_sum = lambda *args: sum(args)
        return splat_sum(1, 2, 3)

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_logical_not(snapshot):
    def f():
        a = True
        b = False
        return not a and not b

    js = compile(f)
    snapshot.assert_match(js, "out.js")


class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class FancyRectangle(Rectangle):
    @property
    def perimeter(self):
        return self.width * 2 + self.height * 2


def test_class(snapshot):
    js = compile(Rectangle)
    snapshot.assert_match(js, "out.js")


def test_class_with_properties(snapshot):
    js = compile(FancyRectangle)
    snapshot.assert_match(js, "out.js")


def test_set_to_object(snapshot):
    def f(a):
        x = set()
        y = 1
        x.add(y)
        return y

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_setitem(snapshot):
    def f(a):
        x = {}
        y = "2"
        x[y] = y
        return x

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_delete(snapshot):
    def f(a):
        x = [a, 1, 2, 3]
        y = {"a": 1}
        del x[0], x[1]
        del x[0 + 3]
        del y.a
        return 1

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_scope_with_while(snapshot):
    def f():
        class Foo:
            def do_stuff(self):
                while True:
                    i = 1
                    i + 1
                    break

                while True:
                    i = 1
                    i + 1
                    break

                return 1

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_list_comp(snapshot):
    def f():
        x = [a + b for a, b in [(1, 2), (3, 4), (5, 6)] if a > 1 if b > 2]
        return x

    js = compile(f)
    snapshot.assert_match(js, "out.js")


@pytest.mark.xfail(raises=NotImplementedError, reason="Not yet implemented")
def test_nested_list_comp(snapshot):
    # TODO(phillipc): This can be done by nesting map calls followed by
    # N - 1 calls to array.reduce(Array.concat), where N is the number of
    # generators in the comprehension.
    def f():
        x = [
            a + b + c
            for a in [1, 4, 7]
            for b in [2, 5, 8]
            for c in [3, 6, 9]
            if a > 1
            if b > 2
            if c > 3
        ]
        return x

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_splat(snapshot):
    def f(x, y, z):
        def g(a, b, c):
            return a - b - c

        args = [x, y, z]
        return g(*args)

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_varargs(snapshot):
    def f(*args):
        return sum(*args)

    js = compile(f)
    snapshot.assert_match(js, "out.js")


def test_missing_vararg(snapshot):
    def my_range(n):
        return [1 for x in [n]]

    js = compile(my_range)
    snapshot.assert_match(js, "out.js")


def test_len_rewrite(snapshot):
    def my_func(a):
        return len(a)

    js = compile(my_func)
    snapshot.assert_match(js, "out.js")
