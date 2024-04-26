SELECT
  ST_ASEWKB("t1"."GeoEndPoint(geo_linestring)") AS "GeoEndPoint(geo_linestring)"
FROM (
  SELECT
    ST_ENDPOINT("t0"."geo_linestring") AS "GeoEndPoint(geo_linestring)"
  FROM "geo" AS "t0"
) AS "t1"