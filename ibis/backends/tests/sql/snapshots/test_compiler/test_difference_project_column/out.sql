SELECT
  "t6"."key"
FROM (
  SELECT
    "t5"."key",
    "t5"."value"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t1"."string_col" AS "key",
        CAST("t1"."float_col" AS DOUBLE) AS "value"
      FROM (
        SELECT
          *
        FROM "functional_alltypes" AS "t0"
        WHERE
          "t0"."int_col" > CAST(0 AS TINYINT)
      ) AS "t1"
    ) AS "t4"
    EXCEPT
    SELECT
      *
    FROM (
      SELECT
        "t2"."string_col" AS "key",
        "t2"."double_col" AS "value"
      FROM (
        SELECT
          *
        FROM "functional_alltypes" AS "t0"
        WHERE
          "t0"."int_col" <= CAST(0 AS TINYINT)
      ) AS "t2"
    ) AS "t3"
  ) AS "t5"
) AS "t6"