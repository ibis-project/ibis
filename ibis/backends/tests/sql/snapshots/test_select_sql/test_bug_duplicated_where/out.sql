SELECT
  *
FROM (
  SELECT
    "t1"."arrdelay",
    "t1"."dest",
    AVG("t1"."arrdelay") OVER (PARTITION BY "t1"."dest" ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "dest_avg",
    "t1"."arrdelay" - AVG("t1"."arrdelay") OVER (PARTITION BY "t1"."dest" ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "dev"
  FROM (
    SELECT
      "t0"."arrdelay",
      "t0"."dest"
    FROM "airlines" AS "t0"
  ) AS "t1"
) AS "t2"
WHERE
  "t2"."dev" IS NOT NULL
ORDER BY
  "t2"."dev" DESC
LIMIT 10