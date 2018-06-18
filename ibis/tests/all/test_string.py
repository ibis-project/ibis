import pytest
from pytest import param

import six
import ibis
import ibis.tests.util as tu
import ibis.expr.datatypes as dt
from ibis.compat import maketrans


def test_string_col_is_unicode(backend, alltypes, df):
    dtype = alltypes.string_col.type()
    assert dtype == dt.String(nullable=dtype.nullable)
    is_text_type = lambda x: isinstance(x, six.text_type)  # noqa: E731
    assert df.string_col.map(is_text_type).all()
    result = alltypes.string_col.execute()
    assert result.map(is_text_type).all()


@pytest.mark.parametrize(
    ('result_func', 'expected_func'),
    [
        param(
            lambda t: t.string_col.contains('6'),
            lambda t: t.string_col.str.contains('6'),
            id='contains',
        ),
        param(
            lambda t: t.string_col.like('6%'),
            lambda t: t.string_col.str.contains('6.*'),
            id='like',
        ),
        param(
            lambda t: t.string_col.like('6^%'),
            lambda t: t.string_col.str.contains('6%'),
            id='complex_like_escape',
        ),
        param(
            lambda t: t.string_col.like('6^%%'),
            lambda t: t.string_col.str.contains('6%.*'),
            id='complex_like_escape_match',
        ),
        param(
            lambda t: t.string_col.re_search(r'[[:digit:]]+'),
            lambda t: t.string_col.str.contains(r'\d+'),
            id='re_search',
        ),
        param(
            lambda t: t.string_col.re_extract(r'([[:digit:]]+)', 0),
            lambda t: t.string_col.str.extract(r'(\d+)', expand=False),
            id='re_extract',
        ),
        param(
            lambda t: t.string_col.re_replace(r'[[:digit:]]+', 'a'),
            lambda t: t.string_col.str.replace(r'\d+', 'a'),
            id='re_replace',
        ),
        param(
            lambda t: t.string_col.repeat(2),
            lambda t: t.string_col * 2,
            id='repeat'
        ),
        param(
            lambda t: t.string_col.translate(u'0', u'a'),
            lambda t: t.string_col.str.translate(maketrans(u'0', u'a')),
            id='translate',
        ),
        param(
            lambda t: t.string_col.find('a'),
            lambda t: t.string_col.str.find('a'),
            id='find'
        ),
        param(
            lambda t: t.string_col.lpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='left'),
            id='lpad'
        ),
        param(
            lambda t: t.string_col.rpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='right'),
            id='rpad',
        ),
        param(
            lambda t: t.string_col.find_in_set(['1']),
            lambda t: t.string_col.str.find('1'),
            id='find_in_set',
        ),
        param(
            lambda t: t.string_col.find_in_set(['a']),
            lambda t: t.string_col.str.find('a'),
            id='find_in_set_all_missing',
        ),
        param(
            lambda t: t.string_col.lower(),
            lambda t: t.string_col.str.lower(),
            id='lower'
        ),
        param(
            lambda t: t.string_col.upper(),
            lambda t: t.string_col.str.upper(),
            id='upper'
        ),
        param(
            lambda t: t.string_col.reverse(),
            lambda t: t.string_col.str[::-1],
            id='reverse'
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            lambda t: t.string_col.map(ord).astype('int32'),
            id='ascii_str'
        ),
        param(
            lambda t: t.string_col.length(),
            lambda t: t.string_col.str.len().astype('int32'),
            id='length',
        ),
        param(
            lambda t: t.string_col.strip(),
            lambda t: t.string_col.str.strip(),
            id='strip'
        ),
        param(
            lambda t: t.string_col.lstrip(),
            lambda t: t.string_col.str.lstrip(),
            id='lstrip'
        ),
        param(
            lambda t: t.string_col.rstrip(),
            lambda t: t.string_col.str.rstrip(),
            id='rstrip'
        ),
        param(
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id='capitalize',
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:5],
            id='substr'
        ),
        param(
            lambda t: t.date_string_col.left(2),
            lambda t: t.date_string_col.str[:2],
            id='left',
        ),
        param(
            lambda t: t.date_string_col.right(2),
            lambda t: t.date_string_col.str[-2:],
            id='right',
        ),
        param(
            lambda t: t.date_string_col.split('/'),
            lambda t: t.date_string_col.str.split('/'),
            id='split',
        ),
        param(
            lambda t: ibis.literal('-').join(['a', t.string_col, 'c']),
            lambda t: 'a-' + t.string_col + '-c',
            id='join'
        )
    ],
)
@tu.skipif_unsupported
def test_string(backend, alltypes, df, result_func, expected_func):
    expr = result_func(alltypes)
    result = expr.execute()

    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)
