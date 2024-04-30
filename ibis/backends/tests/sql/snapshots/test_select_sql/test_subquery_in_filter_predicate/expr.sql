SELECT
  *
FROM "star1" AS "t0"
WHERE
  "t0"."f" > (
    SELECT
      AVG("t0"."f") AS "Mean(f)"
    FROM "star1" AS "t0"
  )