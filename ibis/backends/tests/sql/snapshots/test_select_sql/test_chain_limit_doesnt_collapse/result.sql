SELECT
  *
FROM (
  SELECT
    "t1"."city",
    "t1"."CountStar(tbl)"
  FROM (
    SELECT
      "t0"."city",
      COUNT(*) AS "CountStar(tbl)"
    FROM "tbl" AS "t0"
    GROUP BY
      1
  ) AS "t1"
  ORDER BY
    "t1"."CountStar(tbl)" DESC
  LIMIT 10
) AS "t3"
LIMIT 5
OFFSET (
  SELECT
    COUNT(*) + CAST(-5 AS TINYINT)
  FROM (
    SELECT
      "t1"."city",
      "t1"."CountStar(tbl)"
    FROM (
      SELECT
        "t0"."city",
        COUNT(*) AS "CountStar(tbl)"
      FROM "tbl" AS "t0"
      GROUP BY
        1
    ) AS "t1"
    ORDER BY
      "t1"."CountStar(tbl)" DESC
    LIMIT 10
  ) AS "t3"
)