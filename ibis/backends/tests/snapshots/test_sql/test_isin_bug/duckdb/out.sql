SELECT
  "t0"."x" IN (
    SELECT
      "t0"."x"
    FROM "t" AS "t0"
    WHERE
      "t0"."x" > CAST(2 AS TINYINT)
  ) AS "InSubquery(x)"
FROM "t" AS "t0"