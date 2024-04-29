SELECT
  *
FROM "functional_alltypes" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM "functional_alltypes" AS "t1"
    WHERE
      "t0"."string_col" = "t1"."string_col"
  )