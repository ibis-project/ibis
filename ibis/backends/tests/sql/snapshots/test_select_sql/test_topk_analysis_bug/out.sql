WITH "t1" AS (
  SELECT
    "t0"."dest",
    "t0"."origin",
    "t0"."arrdelay"
  FROM "airlines" AS "t0"
  WHERE
    "t0"."dest" IN ('ORD', 'JFK', 'SFO')
)
SELECT
  "t8"."origin",
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    "t3"."dest",
    "t3"."origin",
    "t3"."arrdelay"
  FROM "t1" AS "t3"
  SEMI JOIN (
    SELECT
      "t4"."dest",
      "t4"."Mean(arrdelay)"
    FROM (
      SELECT
        "t2"."dest",
        AVG("t2"."arrdelay") AS "Mean(arrdelay)"
      FROM "t1" AS "t2"
      GROUP BY
        1
    ) AS "t4"
    ORDER BY
      "t4"."Mean(arrdelay)" DESC
    LIMIT 10
  ) AS "t7"
    ON "t3"."dest" = "t7"."dest"
) AS "t8"
GROUP BY
  1