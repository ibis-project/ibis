SELECT
  *
FROM "star1" AS "t0"
WHERE
  "t0"."f" > 0 AND "t0"."c" < (
    "t0"."f" * 2
  )