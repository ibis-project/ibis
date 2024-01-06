SELECT
  ST_ASEWKB(ST_STARTPOINT("t0"."geo_linestring")) AS "GeoStartPoint(geo_linestring)"
FROM "geo" AS "t0"