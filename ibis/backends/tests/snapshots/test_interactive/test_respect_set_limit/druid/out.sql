SELECT
  *
FROM (
  SELECT
    "t0"."id",
    1 - (
      MOD("t0"."id", 2)
    ) AS "bool_col"
  FROM "functional_alltypes" AS "t0"
  LIMIT 10
) AS "t2"
LIMIT 11