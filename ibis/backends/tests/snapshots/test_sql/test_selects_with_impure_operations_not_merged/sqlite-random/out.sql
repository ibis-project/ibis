SELECT
  "t1"."x",
  "t1"."y",
  "t1"."z",
  IIF("t1"."y" = "t1"."z", 'big', 'small') AS "size"
FROM (
  SELECT
    "t0"."x",
    0.5 + (
      CAST(RANDOM() AS REAL) / -1.8446744073709552e+19
    ) AS "y",
    0.5 + (
      CAST(RANDOM() AS REAL) / -1.8446744073709552e+19
    ) AS "z"
  FROM "t" AS "t0"
) AS "t1"