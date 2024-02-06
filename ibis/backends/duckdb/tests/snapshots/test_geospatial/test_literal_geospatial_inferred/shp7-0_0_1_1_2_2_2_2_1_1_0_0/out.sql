SELECT
  ST_ASWKB("result") AS result
FROM (
  SELECT
    ST_GEOMFROMTEXT('MULTILINESTRING ((0 0, 1 1, 2 2), (2 2, 1 1, 0 0))') AS "result"
)