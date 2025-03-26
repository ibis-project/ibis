SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t0"."city",
      COUNT(*) AS "city_count"
    FROM "tbl" AS "t0"
    GROUP BY
      1
  ) AS "t1"
  ORDER BY
    "t1"."city_count" DESC
  LIMIT 10
) AS "t3"
LIMIT 5
OFFSET (
  SELECT
    COUNT(*) + -5
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t0"."city",
        COUNT(*) AS "city_count"
      FROM "tbl" AS "t0"
      GROUP BY
        1
    ) AS "t1"
    ORDER BY
      "t1"."city_count" DESC
    LIMIT 10
  ) AS "t3"
)