SELECT
  "t1"."foo",
  "t1"."bar",
  "t1"."city",
  "t1"."v1",
  "t1"."v2"
FROM "tbl" AS "t1"
SEMI JOIN (
  SELECT
    *
  FROM (
    SELECT
      "t0"."city",
      COUNT(*) AS "city_count"
    FROM "tbl" AS "t0"
    GROUP BY
      1
  ) AS "t2"
  ORDER BY
    "t2"."city_count" DESC
  LIMIT 10
) AS "t5"
  ON "t1"."city" = "t5"."city"