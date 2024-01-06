SELECT
  ST_CONTAINS("t0"."geo_linestring", "t0"."geo_point") AS "GeoContains(geo_linestring, geo_point)"
FROM "geo" AS "t0"