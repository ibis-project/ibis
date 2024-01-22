SELECT
  "t0"."x" IN (
    SELECT
      "t0"."x"
    FROM "t" "t0"
    WHERE
      "t0"."x" > 2
  ) AS "InSubquery(x)"
FROM "t" "t0"