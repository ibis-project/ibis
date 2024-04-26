SELECT
  *
FROM (
  SELECT
    "t7"."a"
  FROM (
    SELECT
      "t5"."a",
      "t5"."b",
      MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0) AS "the_date"
    FROM (
      SELECT
        "t4"."a",
        "t4"."b"
      FROM (
        SELECT
          *
        FROM (
          SELECT
            "t1"."a",
            "t1"."b",
            "t1"."c" AS "C"
          FROM "t" AS "t1"
        ) AS "t3"
        WHERE
          "t3"."C" = MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0)
      ) AS "t4"
    ) AS "t5"
  ) AS "t7"
  INNER JOIN "s" AS "t2"
    ON "t7"."b" = "t2"."b"
) AS "t8"
WHERE
  "t8"."a" < CAST(1.0 AS DOUBLE)