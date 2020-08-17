#!/bin/sh
#
# Create SQLite database for tests

rm -f $IBIS_TEST_SQLITE_DATABASE

sqlite3 $IBIS_TEST_SQLITE_DATABASE < tests_schema.sql

for TABLE in awards_players diamonds batting functional_alltypes; do
    echo "Loading $TABLE..."

    sqlite3 $IBIS_TEST_SQLITE_DATABASE <<-EOF
	.mode csv
	.import $IBIS_TEST_DATA_DIRECTORY/$TABLE.csv $TABLE --skip 1
	EOF
done
