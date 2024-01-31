SELECT
  "t1"."foo",
  "t1"."bar",
  "t1"."city",
  "t1"."v1",
  "t1"."v2"
FROM "tbl" AS "t1"
SEMI JOIN (
  SELECT
    "t2"."city",
    "t2"."Count(city)"
  FROM (
    SELECT
      "t0"."city",
      COUNT("t0"."city") AS "Count(city)"
    FROM "tbl" AS "t0"
    GROUP BY
      1
  ) AS "t2"
  ORDER BY
    "t2"."Count(city)" DESC
  LIMIT 10
) AS "t5"
  ON "t1"."city" = "t5"."city"