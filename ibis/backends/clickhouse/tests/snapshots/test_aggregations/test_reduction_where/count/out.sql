SELECT
  countIf(t0.double_col, t0.bigint_col < 70) AS "Count(double_col, Less(bigint_col, 70))"
FROM functional_alltypes AS t0