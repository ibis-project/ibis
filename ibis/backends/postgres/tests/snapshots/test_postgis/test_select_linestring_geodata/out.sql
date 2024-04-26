SELECT
  ST_ASEWKB("t1"."geo_linestring") AS "geo_linestring"
FROM (
  SELECT
    "t0"."geo_linestring"
  FROM "geo" AS "t0"
) AS "t1"