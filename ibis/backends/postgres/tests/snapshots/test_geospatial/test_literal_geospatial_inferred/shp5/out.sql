SELECT
  ST_ASEWKB("t0"."<POLYGON ((0 0, 1 1, 2 2, 0 0))>") AS "<POLYGON ((0 0, 1 1, 2 2, 0 0))>"
FROM (
  SELECT
    ST_GEOMFROMTEXT('POLYGON ((0 0, 1 1, 2 2, 0 0))') AS "<POLYGON ((0 0, 1 1, 2 2, 0 0))>"
) AS "t0"