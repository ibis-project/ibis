SELECT
  ST_ASEWKB("t0"."<MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))>") AS "<MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))>"
FROM (
  SELECT
    ST_GEOMFROMTEXT('MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))') AS "<MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))>"
) AS "t0"