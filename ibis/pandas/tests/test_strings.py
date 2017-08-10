import pytest
from warnings import catch_warnings
import pandas.util.testing as tm  # noqa: E402


@pytest.mark.parametrize(
    ('case_func', 'expected_func'),
    [
        (lambda s: s.length(), lambda s: s.str.len()),
        (lambda s: s.substr(1, 2), lambda s: s.str[1:3]),
        (lambda s: s.strip(), lambda s: s.str.strip()),
        (lambda s: s.lstrip(), lambda s: s.str.lstrip()),
        (lambda s: s.rstrip(), lambda s: s.str.rstrip()),
        (
            lambda s: s.lpad(3, 'a'),
            lambda s: s.str.pad(3, side='left', fillchar='a')
        ),
        (
            lambda s: s.rpad(3, 'b'),
            lambda s: s.str.pad(3, side='right', fillchar='b')
        ),
        (lambda s: s.reverse(), lambda s: s.str[::-1]),
        (lambda s: s.lower(), lambda s: s.str.lower()),
        (lambda s: s.upper(), lambda s: s.str.upper()),
        (lambda s: s.capitalize(), lambda s: s.str.capitalize()),
        (lambda s: s.repeat(2), lambda s: s * 2),
        (
            lambda s: s.contains('a'),
            lambda s: s.str.contains('a', regex=False)
        ),
        (
            lambda s: ~s.contains('a'),
            lambda s: ~s.str.contains('a', regex=False)
        ),
        (
            lambda s: s.like('a'),
            lambda s: s.str.contains('a', regex=True),
        ),
        (
            lambda s: s.like('(ab)+'),
            lambda s: s.str.contains('(ab)+', regex=True),
        ),
        (
            lambda s: s.like(['(ab)+', 'd{1,2}ee']),
            lambda s: (
                s.str.contains('(ab)+', regex=True) |
                s.str.contains('d{1,2}ee')
            )
        ),
        pytest.mark.xfail(
            (
                lambda s: s + s.rpad(3, 'a'),
                lambda s: s + s.str.pad(3, side='right', fillchar='a')
            ),
            raises=NotImplementedError,
            reason='Implement string concat with plus'
        ),
    ]
)
def test_string_ops(t, df, case_func, expected_func):

    # ignore matching UserWarnings
    with catch_warnings(record=True):
        expr = case_func(t.strings_with_space)
        result = expr.execute()
        series = expected_func(df.strings_with_space)
        tm.assert_series_equal(result, series)
