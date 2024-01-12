SELECT
  "t4"."id",
  "t4"."tz",
  "t4"."no_tz",
  "t2"."id" AS "id_right"
FROM (
  SELECT
    "t0"."id",
    "t0"."ts_tz" AS "tz",
    "t0"."ts_no_tz" AS "no_tz"
  FROM "x" AS "t0"
) AS "t4"
LEFT OUTER JOIN "y" AS "t2"
  ON "t4"."id" = "t2"."id"