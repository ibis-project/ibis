SELECT
  "t0"."x" IN (
    SELECT
      *
    FROM "t" AS "t0"
    WHERE
      "t0"."x" > 2
  ) AS "InSubquery(x)"
FROM "t" AS "t0"