SELECT
  ST_ASWKB("result") AS result
FROM (
  SELECT
    ST_GEOMFROMTEXT('MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))') AS "result"
)