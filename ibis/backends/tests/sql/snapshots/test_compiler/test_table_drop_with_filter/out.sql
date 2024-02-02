SELECT
  "t5"."a"
FROM (
  SELECT
    "t4"."a"
  FROM (
    SELECT
      "t0"."a",
      "t0"."b",
      MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0) AS "the_date"
    FROM "t" AS "t0"
    WHERE
      "t0"."c" = MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0)
  ) AS "t4"
  INNER JOIN "s" AS "t2"
    ON "t4"."b" = "t2"."b"
) AS "t5"
WHERE
  "t5"."a" < CAST(1.0 AS DOUBLE)