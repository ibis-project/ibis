SELECT
  "t1"."arrdelay",
  "t1"."dest",
  "t1"."dest_avg",
  "t1"."dev"
FROM (
  SELECT
    "t0"."arrdelay",
    "t0"."dest",
    AVG("t0"."arrdelay") OVER (PARTITION BY "t0"."dest" ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "dest_avg",
    "t0"."arrdelay" - AVG("t0"."arrdelay") OVER (PARTITION BY "t0"."dest" ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "dev"
  FROM "airlines" AS "t0"
) AS "t1"
WHERE
  "t1"."dev" IS NOT NULL
ORDER BY
  "t1"."dev" DESC
LIMIT 10