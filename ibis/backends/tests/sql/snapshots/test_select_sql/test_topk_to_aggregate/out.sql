SELECT
  "t1"."dest",
  "t1"."Mean(arrdelay)"
FROM (
  SELECT
    "t0"."dest",
    AVG("t0"."arrdelay") AS "Mean(arrdelay)"
  FROM "airlines" AS "t0"
  GROUP BY
    1
) AS "t1"
ORDER BY
  "t1"."Mean(arrdelay)" DESC
LIMIT 10