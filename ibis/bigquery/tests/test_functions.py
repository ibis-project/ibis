import pytest


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_arbitrary(alltypes):
    """Test that the value returned is one of the values possible
    """
    possible_months = (
        alltypes
        .groupby([alltypes.year, alltypes.month]).count()
        .groupby('year')
        .aggregate(months=lambda t: t.month.collect())
        .execute()
    )
    result = (
        alltypes
        .groupby([alltypes.year])
        .aggregate(arbitrary_month=lambda t: t.month.arbitrary())
        .execute()
    )
    assert (
        possible_months
        .merge(result)
        .apply(lambda row: row.arbitrary_month in row.months, axis=1)
        .all()
    )
