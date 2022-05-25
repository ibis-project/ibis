import pandas.testing as tm
import pytest


@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_single_field(backend, struct, struct_df, field):
    result = struct.abc[field].execute()
    expected = struct_df.abc.map(
        lambda value: value[field] if isinstance(value, dict) else value
    ).rename(field)
    backend.assert_series_equal(result, expected)


def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc
    tm.assert_series_equal(result, expected)
