import pandas.testing as tm
import pytest
from pytest import param


@pytest.mark.parametrize(
    ('result_func', 'expected_func'),
    [
        param(
            lambda t: t.date_string_col.re_extract(r'(\d+)\D(\d+)\D(\d+)', 1),
            lambda t: t.date_string_col.str.extract(
                r'(\d+)\D(\d+)\D(\d+)', expand=False
            ).iloc[:, 0],
            id='re_extract_group_1',
        ),
        param(
            lambda t: t.date_string_col.re_extract(r'(\d+)\D(\d+)\D(\d+)', 2),
            lambda t: t.date_string_col.str.extract(
                r'(\d+)\D(\d+)\D(\d+)', expand=False
            ).iloc[:, 1],
            id='re_extract_group_2',
        ),
        param(
            lambda t: t.date_string_col.re_extract(r'(\d+)\D(\d+)\D(\d+)', 3),
            lambda t: t.date_string_col.str.extract(
                r'(\d+)\D(\d+)\D(\d+)', expand=False
            ).iloc[:, 2],
            id='re_extract_group_3',
        ),
        param(
            lambda t: t.date_string_col.re_extract(r'^(\d+)', 1),
            lambda t: t.date_string_col.str.extract(r'^(\d+)', expand=False),
            id='re_extract_group_at_beginning',
        ),
        param(
            lambda t: t.date_string_col.re_extract(r'(\d+)$', 1),
            lambda t: t.date_string_col.str.extract(r'(\d+)$', expand=False),
            id='re_extract_group_at_end',
        ),
    ],
)
def test_string(alltypes, df, result_func, expected_func):
    expr = result_func(alltypes).name('tmp')
    result = expr.execute()

    expected = expected_func(df)
    tm.assert_series_equal(result, expected, check_names=False)
