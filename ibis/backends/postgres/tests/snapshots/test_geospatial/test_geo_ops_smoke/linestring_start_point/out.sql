SELECT
  ST_ASEWKB("t1"."GeoStartPoint(geo_linestring)") AS "GeoStartPoint(geo_linestring)"
FROM (
  SELECT
    ST_STARTPOINT("t0"."geo_linestring") AS "GeoStartPoint(geo_linestring)"
  FROM "geo" AS "t0"
) AS "t1"