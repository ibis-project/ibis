import pytest

import ibis.expr.datatypes as dt

ibis_type_to_bigquery_type = pytest.importorskip(
    'ibis.bigquery.udf.core.ibis_type_to_bigquery_type'
)
SymbolTable = pytest.importorskip('ibis.bigquery.udf.core.SymbolTable')
PythonToJavaScriptTranslator = pytest.importorskip(
    'ibis.bigquery.udf.core.PythonToJavaScriptTranslator'
)


@pytest.mark.parametrize(
    ('type', 'expected'),
    [
        (dt.float64, 'FLOAT64'),
        (dt.int64, 'FLOAT64'),  # JavaScript only has a Number type
        (dt.string, 'STRING'),
        (dt.Array(dt.int64), 'ARRAY<FLOAT64>'),
        (dt.Array(dt.string), 'ARRAY<STRING>'),
        (
            dt.Struct.from_tuples([
                ('a', dt.int64),
                ('b', dt.string),
                ('c', dt.Array(dt.string)),
            ]),
            'STRUCT<a: INT64, b: STRING, c: ARRAY<STRING>>'
        ),
        (dt.date, 'DATE'),
        (dt.timestamp, 'TIMESTAMP'),
        pytest.mark.xfail(
            (dt.timestamp(timezone='US/Eastern'), 'TIMESTAMP'),
            raises=TypeError,
            reason='Not supported in BigQuery'
        ),
        ('array<struct<a: string>>', 'ARRAY<STRUCT<a: STRING>>'),
    ]
)
def test_ibis_type_to_bigquery_type(type, expected):
    result = ibis_type_to_bigquery_type(type)
    assert result == expected


def test_symbol_table():
    symbols = SymbolTable()
    assert symbols['a'] == 'let a'
    assert symbols['a'] == 'a'


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


def test_yield():
    def f(a):
        for i in [1, 2, 3]:
            yield i

    js = compile(f)
    expected = """\
function* f(a) {
    for (let i of [1, 2, 3]) {
        yield i;
    }
}"""
    assert expected == js


def test_yield_from():
    def f(a):
        yield from [1, 2, 3]

    js = compile(f)
    expected = """\
function* f(a) {
    yield* [1, 2, 3];
}"""
    assert expected == js


def test_assign():
    def f():
        a = 1
        a = 2
        print(a)
        return 1

    js = compile(f)
    expected = """\
function f() {
    let a = 1;
    a = 2;
    console.log(a);
    return 1;
}"""
    assert expected == js


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (lambda a, b: a + b, '+'),
        (lambda a, b: a - b, '-'),
        (lambda a, b: a * b, '*'),
        (lambda a, b: a / b, '/'),
    ]
)
def test_binary_operators(op, expected):
    js = compile(op)
    expected = """\
function op(a, b) {
    return (a {} b);
}""".format(expected)
    assert expected == js


def test_pow():
    def f():
        a = 1
        return a ** 2

    expected = """\
function f() {
    let a = 1;
    return Math.pow(a, 2);
}"""

    js = compile(f)
    assert expected == js


def test_floordiv():
    def f():
        a = 1
        return a // 2

    expected = """\
function f() {
    let a = 1;
    return Math.floor(a / 2);
}"""

    js = compile(f)
    assert expected == js


def test_unary_minus():
    def f():
        a = 1
        return -a

    expected = """\
function f() {
    let a = 1;
    return (-a);
}"""

    js = compile(f)
    assert expected == js


def test_unary_plus():
    def f():
        a = 1
        return +a

    expected = """\
function f() {
    let a = 1;
    return (+a);
}"""

    js = compile(f)
    assert expected == js


def test_true_false_none():
    def f():
        a = True
        b = False
        c = None
        return a if c != None else b  # noqa: E711
    expected = """\
function f() {
    let a = true;
    let b = false;
    let c = null;
    return c !== null ? a : b;
}"""
    js = compile(f)
    assert expected == js


def test_str():
    def f():
        a = 'abc'
        return a

    expected = """\
function f() {
    let a = 'abc';
    return a;
}"""
    js = compile(f)
    assert expected == js


def test_tuple():
    def f():
        a = 'a', 'b', 'c'
        return a

    expected = """\
function f() {
    let a = ['a', 'b', 'c'];
    return a;
}"""
    js = compile(f)
    assert expected == js


def test_dict():
    def f():
        a = {'a': 1, 'b': 2}
        return a

    expected = """\
function f() {
    let a = {['a']: 1, ['b']: 2};
    return a;
}"""
    js = compile(f)
    assert expected == js


def test_continue():
    def f():
        i = 0
        for i in [1, 2, 3]:
            if i == 1:
                continue
        return i

    expected = """\
function f() {
    let i = 0;
    for (let i of [1, 2, 3]) {
        if (i === 1) {
            continue;
        }
    }
    return i;
}"""
    js = compile(f)
    assert expected == js


def test_lambda():
    def f():
        a = lambda *args: sum(args)  # noqa: E731
        return a(1, 2, 3)

    expected = """\
function f() {
    let a = (...args) => sum(args);
    return a(1, 2, 3);
}"""
    js = compile(f)
    assert expected == js


def test_logical_not():
    def f():
        a = True
        b = False
        return not a and not b

    expected = """\
function f() {
    let a = true;
    let b = false;
    return !a && !b;
}"""
    js = compile(f)
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
