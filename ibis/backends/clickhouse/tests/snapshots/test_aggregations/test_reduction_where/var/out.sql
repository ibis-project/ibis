SELECT
  varSampIf(t0.double_col, t0.bigint_col < 70) AS "Variance(double_col, Less(bigint_col, 70))"
FROM functional_alltypes AS t0