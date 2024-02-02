SELECT
  "t4"."foo_id",
  "t4"."total",
  "t2"."value1"
FROM (
  SELECT
    "t0"."foo_id",
    SUM("t0"."f") AS "total"
  FROM "star1" AS "t0"
  GROUP BY
    1
) AS "t4"
INNER JOIN "star2" AS "t2"
  ON "t4"."foo_id" = "t2"."foo_id"