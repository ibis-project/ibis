import pytest

import ibis
import ibis.expr.lineage as lin

from ibis.tests.util import assert_equal


@pytest.fixture
def companies(con):
    schema = [
        ('permalink', 'string'),
        ('name', 'string'),
        ('homepage_url', 'string'),
        ('category_list', 'string'),
        ('market', 'string'),
        ('funding_total_usd', 'float'),
        ('status', 'string'),
        ('country_code', 'string'),
        ('state_code', 'string'),
        ('region', 'string'),
        ('city', 'string'),
        ('funding_rounds', 'int32'),
        ('founded_at', 'string'),
        ('founded_month', 'string'),
        ('founded_quarter', 'string'),
        ('founded_year', 'float'),
        ('first_funding_at', 'string'),
        ('last_funding_at', 'string'),
    ]
    return ibis.table(schema, name='companies')


@pytest.fixture
def rounds(con):
    schema = [
        ('company_permalink', 'string'),
        ('company_name', 'string'),
        ('company_category_list', 'string'),
        ('company_market', 'string'),
        ('company_country_code', 'string'),
        ('company_state_code', 'string'),
        ('company_region', 'string'),
        ('company_city', 'string'),
        ('funding_round_permalink', 'string'),
        ('funding_round_type', 'string'),
        ('funding_round_code', 'string'),
        ('funded_at', 'string'),
        ('funded_month', 'string'),
        ('funded_quarter', 'string'),
        ('funded_year', 'int32'),
        ('raised_amount_usd', 'float'),
    ]
    return ibis.table(schema, name='rounds')


def test_lineage(companies):
    # single table dependency
    funding_buckets = [
        0, 1000000, 10000000, 50000000, 100000000,  500000000, 1000000000
    ]

    bucket_names = [
        '0 to 1m',
        '1m to 10m',
        '10m to 50m',
        '50m to 100m',
        '100m to 500m',
        '500m to 1b',
        'Over 1b',
    ]

    bucket = (
        companies.funding_total_usd.bucket(funding_buckets, include_over=True)
    )

    mutated = companies.mutate(
        bucket=bucket,
        status=companies.status.fillna('Unknown')
    )

    filtered = mutated[
        (companies.founded_at > '2010-01-01') | companies.founded_at.isnull()
    ]

    grouped = filtered.group_by(['bucket', 'status']).size()

    # TODO(cpcloud): Should this be used?
    joined = grouped.mutate(  # noqa
        bucket_name=lambda x: x.bucket.label(bucket_names).fillna('Unknown')
    )

    results = list(lin.lineage(bucket))
    expected = [
        bucket,
        companies.funding_total_usd,
        companies,
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)

    results = list(lin.lineage(mutated.bucket))
    expected = [
        mutated.bucket,
        mutated,
        bucket.name('bucket'),
        companies.funding_total_usd,
        companies,
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)

    results = list(lin.lineage(filtered.bucket))
    expected = [
        filtered.bucket,
        filtered,
        bucket.name('bucket'),
        companies.funding_total_usd,
        companies,
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)

    results = list(lin.lineage(grouped.bucket))
    expected = [
        grouped.bucket,
        grouped,
        filtered.bucket,
        filtered,
        bucket.name('bucket'),
        companies.funding_total_usd,
        companies
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)


def test_lineage_multiple_parents(companies):
    funding_per_year = companies.funding_total_usd / companies.funding_rounds
    results = list(lin.lineage(funding_per_year))
    expected = [
        funding_per_year,
        companies.funding_total_usd,
        companies,
        companies.funding_rounds,
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)

    # breadth first gives a slightly more aesthetically pleasing result
    results = list(lin.lineage(funding_per_year, container=lin.Queue))
    expected = [
        funding_per_year,
        companies.funding_total_usd,
        companies.funding_rounds,
        companies,
    ]
    for r, e in zip(results, expected):
        assert_equal(r, e)


def test_lineage_join(companies, rounds):
    joined = companies.join(
        rounds,
        companies.first_funding_at.cast(
            'timestamp'
        ).year() == rounds.funded_year
    )
    expr = joined[
        companies.funding_total_usd,
        rounds.funding_round_type,
        rounds.company_city,
        rounds.raised_amount_usd,
    ]
    perc_raised = (
        expr.raised_amount_usd / expr.funding_total_usd
    ).name('perc_raised')
    results = list(lin.lineage(perc_raised))

    expected = [
        perc_raised,
        expr.raised_amount_usd,
        expr,
        rounds.raised_amount_usd,
        rounds,
        expr.funding_total_usd,
        # expr,  # *could* appear here as well, but we've already traversed it
        companies.funding_total_usd,
        companies
    ]
    assert len(results) == len(expected)
    for r, e in zip(results, expected):
        assert_equal(r, e)

    results = list(lin.lineage(perc_raised, container=lin.Queue))
    expected = [
        perc_raised,
        expr.raised_amount_usd,
        expr.funding_total_usd,
        expr,
        rounds.raised_amount_usd,
        companies.funding_total_usd,
        rounds,
        companies
    ]
    assert len(results) == len(expected)
    for r, e in zip(results, expected):
        assert_equal(r, e)
