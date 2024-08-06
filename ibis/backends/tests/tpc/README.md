# TPC queries with Ibis

These tests perform correctness tests against backends that are able to run
some of the TPC-H and TPC-DS queries.

The text queries are assumed to be correct, and also that if transpiled
correctly will produce the same results as the written Ibis expression.

**This is the assertion being made in these tests.**

The ground truth SQL text is taken from
[DuckDB](https://github.com/duckdb/duckdb/tree/main/extension/tpcds/dsdgen/queries)
and transpiled using SQLGlot to the dialect of whatever backend is under test.

Some queries are altered from the upstream DucKDB repo to have static column
names and to cast strings that are dates explicitly to dates so that pedantic
engines like Trino will accept these queries. These alterations do not change
the computed results of the queries.

ClickHouse is a bit odd in that queries that contain a cross join with an `OR`
condition common to all operands of the `OR` will effectively never finish.
This is probably a bug in ClickHouse.

For that case, the queries for clickhouse have been minimally rewritten to pass
by extracting the common join condition out into a single `AND` operand.
