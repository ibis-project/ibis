import pytest

PythonToJavaScriptTranslator = pytest.importorskip(
    'ibis.bigquery.udf.core.PythonToJavaScriptTranslator'
)


def compile(f):
    return PythonToJavaScriptTranslator(f).compile()


def test_function_def():
    def f(a, b):
        return a + b

    expected = """\
function f(a, b) {
    return (a + b);
}"""

    js = compile(f)
    assert expected == js


def test_variable_declaration():
    def f():
        c = 1
        return c + 2

    js = compile(f)

    expected = """\
function f() {
    let c = 1;
    return (c + 2);
}"""
    assert expected == js


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


def test_class():
    expected = """\
class Rectangle {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }
    area() {
        return (this.width * this.height);
    }
}"""
    js = compile(Rectangle)
    assert expected == js


def test_class_with_properties():
    expected = """\
class FancyRectangle extends Rectangle {
    get perimeter() {
        return ((this.width * 2) + (this.height * 2));
    }
}"""
    js = compile(FancyRectangle)
    assert expected == js


def test_set_to_object():
    def f(a):
        x = set()
        y = 1
        x.add(y)
        return y

    expected = """\
function f(a) {
    let x = (new Set());
    let y = 1;
    x.add(y);
    return y;
}"""
    js = compile(f)
    assert expected == js


def test_setitem():
    def f(a):
        x = {}
        y = '2'
        x[y] = y
        return x
    expected = """\
function f(a) {
    let x = {};
    let y = '2';
    x[y] = y;
    return x;
}"""
    js = compile(f)
    assert expected == js


def test_delete():
    def f(a):
        x = [a, 1, 2, 3]
        y = {'a': 1}
        del x[0], x[1]
        del x[0 + 3]
        del y.a
        return 1
    expected = """\
function f(a) {
    let x = [a, 1, 2, 3];
    let y = {['a']: 1};
    delete x[0];
    delete x[1];
    delete x[(0 + 3)];
    delete y.a;
    return 1;
}"""
    js = compile(f)
    assert expected == js


def test_scope_with_while():
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

    expected = """\
function f() {
    class Foo {
        do_stuff() {
            while (true) {
                let i = 1;
                (i + 1);
                break;
            }
            while (true) {
                let i = 1;
                (i + 1);
                break;
            }
            return 1;
        }
    }
}"""
    js = compile(f)
    assert expected == js


def test_list_comp():
    def f():
        x = [a + b for a, b in [(1, 2), (3, 4), (5, 6)] if a > 1 if b > 2]
        return x
    expected = """\
function f() {
    let x = [[1, 2], [3, 4], [5, 6]].filter((([a, b]) => ((a > 1) && (b > 2)))).map((([a, b]) => (a + b)));
    return x;
}"""  # noqa: E501
    js = compile(f)
    assert expected == js


@pytest.mark.xfail(raises=NotImplementedError, reason='Not yet implemented')
def test_nested_list_comp():
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
    expected = """\
function f() {
    let x = [1, 4, 7].map(
        a => [2, 5, 8].map(
            b => [3, 6, 9].filter(
                c => a > 1 && b > 2 && c > 3
            ).map(c => a + b + c)
        )
    );
    return x;
}"""
    js = compile(f)
    assert js == expected


def test_splat():
    def f(x, y, z):
        def g(a, b, c):
            return a - b - c
        args = [x, y, z]
        return g(*args)
    expected = """\
function f(x, y, z) {
    function g(a, b, c) {
        return ((a - b) - c);
    }
    let args = [x, y, z];
    return g(...args);
}"""
    js = compile(f)
    assert js == expected


def test_varargs():
    def f(*args):
        return sum(*args)
    expected = """\
function f(...args) {
    return sum(...args);
}"""
    js = compile(f)
    assert js == expected
