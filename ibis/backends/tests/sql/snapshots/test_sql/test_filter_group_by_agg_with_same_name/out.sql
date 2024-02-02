SELECT
  "t1"."int_col",
  "t1"."bigint_col"
FROM (
  SELECT
    "t0"."int_col",
    SUM("t0"."bigint_col") AS "bigint_col"
  FROM "t" AS "t0"
  GROUP BY
    1
) AS "t1"
WHERE
  "t1"."bigint_col" = CAST(60 AS TINYINT)