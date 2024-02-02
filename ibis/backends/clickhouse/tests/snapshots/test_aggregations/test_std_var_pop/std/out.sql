SELECT
  stddevPopIf("t0"."double_col", "t0"."bigint_col" < 70) AS "StandardDev(double_col, Less(bigint_col, 70))"
FROM "functional_alltypes" AS "t0"