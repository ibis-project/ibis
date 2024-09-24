SELECT
  *
FROM (
  SELECT
    *
  FROM "test" AS "t0"
  WHERE
    "t0"."x" > 10
) AS "t1"
WHERE
  (
    0.5 + (
      CAST(RANDOM() AS REAL) / -1.8446744073709552e+19
    )
  ) <= 0.5