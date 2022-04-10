import pytest

from ibis import literal as L
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(lambda s: s.lower(), 'lower(`string_col`)', id="lower"),
        pytest.param(lambda s: s.upper(), 'upper(`string_col`)', id="upper"),
        pytest.param(
            lambda s: s.reverse(),
            'reverse(`string_col`)',
            id="reverse",
        ),
        pytest.param(lambda s: s.strip(), 'trim(`string_col`)', id="strip"),
        pytest.param(lambda s: s.lstrip(), 'ltrim(`string_col`)', id="lstrip"),
        pytest.param(lambda s: s.rstrip(), 'rtrim(`string_col`)', id="rstrip"),
        pytest.param(
            lambda s: s.capitalize(),
            'initcap(`string_col`)',
            id="capitalize",
        ),
        pytest.param(
            lambda s: s.length(),
            'length(`string_col`)',
            id="length",
        ),
        pytest.param(
            lambda s: s.ascii_str(),
            'ascii(`string_col`)',
            id="ascii_str",
        ),
        pytest.param(
            lambda s: s.substr(2),
            'substr(`string_col`, 2 + 1)',
            id="substr_2",
        ),
        pytest.param(
            lambda s: s.substr(0, 3),
            'substr(`string_col`, 0 + 1, 3)',
            id="substr_0_3",
        ),
        pytest.param(
            lambda s: s.right(4),
            'strright(`string_col`, 4)',
            id="strright",
        ),
        pytest.param(
            lambda s: s.like('foo%'),
            "`string_col` LIKE 'foo%'",
            id="like",
        ),
        pytest.param(
            lambda s: s.like(['foo%', '%bar']),
            "`string_col` LIKE 'foo%' OR `string_col` LIKE '%bar'",
            id="like_multiple",
        ),
        pytest.param(
            lambda s: s.rlike(r'[\d]+'),
            r"regexp_like(`string_col`, '[\d]+')",
            id="rlike",
        ),
        pytest.param(
            lambda s: s.re_search(r'[\d]+'),
            r"regexp_like(`string_col`, '[\d]+')",
            id="re_search",
        ),
        pytest.param(
            lambda s: s.re_extract(r'[\d]+', 0),
            r"regexp_extract(`string_col`, '[\d]+', 0)",
            id="re_extract",
        ),
        pytest.param(
            lambda s: s.re_replace(r'[\d]+', 'aaa'),
            r"regexp_replace(`string_col`, '[\d]+', 'aaa')",
            id="re_replace",
        ),
        pytest.param(
            lambda s: s.repeat(2),
            'repeat(`string_col`, 2)',
            id="repeat",
        ),
        pytest.param(
            lambda s: s.parse_url('HOST'),
            "parse_url(`string_col`, 'HOST')",
            id="parse_url",
        ),
        pytest.param(
            lambda s: s.translate('a', 'b'),
            "translate(`string_col`, 'a', 'b')",
            id="translate",
        ),
        pytest.param(
            lambda s: s.find('a'),
            "locate('a', `string_col`) - 1",
            id="find",
        ),
        pytest.param(
            lambda s: s.find('a', 2),
            "locate('a', `string_col`, 3) - 1",
            id="find_with_offset",
        ),
        pytest.param(
            lambda s: s.lpad(1, 'a'),
            "lpad(`string_col`, 1, 'a')",
            id="lpad_char",
        ),
        pytest.param(
            lambda s: s.lpad(25),
            "lpad(`string_col`, 25, ' ')",
            id="lpad_default",
        ),
        pytest.param(
            lambda s: s.rpad(1, 'a'),
            "rpad(`string_col`, 1, 'a')",
            id="rpad_char",
        ),
        pytest.param(
            lambda s: s.rpad(25),
            "rpad(`string_col`, 25, ' ')",
            id="rpad_default",
        ),
        pytest.param(
            lambda s: s.find_in_set(['a']),
            "find_in_set(`string_col`, 'a') - 1",
            id="find_in_set_single",
        ),
        pytest.param(
            lambda s: s.find_in_set(['a', 'b']),
            "find_in_set(`string_col`, 'a,b') - 1",
            id="find_in_set_multiple",
        ),
    ],
)
def test_string_builtins(table, expr_fn, expected):
    expr = expr_fn(table.string_col)
    assert translate(expr) == expected


def test_find(table):
    expr = table.string_col.find('a', start=table.tinyint_col)
    expected = "locate('a', `string_col`, `tinyint_col` + 1) - 1"
    assert translate(expr) == expected


def test_string_join():
    expr = L(',').join(['a', 'b'])
    expected = "concat_ws(',', 'a', 'b')"
    assert translate(expr) == expected
