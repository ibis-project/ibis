SELECT
  ST_ASEWKB("t1"."geo_multipolygon") AS "geo_multipolygon"
FROM (
  SELECT
    "t0"."geo_multipolygon"
  FROM "geo" AS "t0"
) AS "t1"