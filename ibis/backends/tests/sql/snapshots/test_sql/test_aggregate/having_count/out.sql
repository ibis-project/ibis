SELECT
  "t1"."foo_id",
  "t1"."total"
FROM (
  SELECT
    "t0"."foo_id",
    SUM("t0"."f") AS "total",
    COUNT(*) AS "CountStar()"
  FROM "star1" AS "t0"
  GROUP BY
    1
) AS "t1"
WHERE
  "t1"."CountStar()" > CAST(100 AS TINYINT)