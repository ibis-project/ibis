SELECT
  "t0"."x" IN (
    SELECT
      *
    FROM "t" AS "t0"
    WHERE
      "t0"."x" > CAST(2 AS TINYINT)
  ) AS "InSubquery(x)"
FROM "t" AS "t0"