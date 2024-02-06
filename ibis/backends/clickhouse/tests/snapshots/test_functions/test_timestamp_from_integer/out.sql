SELECT
  toDateTime("t0"."int_col") AS "TimestampFromUNIX(int_col, SECOND)"
FROM "functional_alltypes" AS "t0"