SELECT
  ST_ASEWKB("t1"."geo_polygon") AS "geo_polygon"
FROM (
  SELECT
    "t0"."geo_polygon"
  FROM "geo" AS "t0"
) AS "t1"