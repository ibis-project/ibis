SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t0"."id",
      CAST("t0"."tinyint_col" AS BIGINT) AS "i",
      CAST(CAST("t0"."string_col" AS TEXT) AS TEXT) AS "s"
    FROM "functional_alltypes" AS "t0"
  ) AS "t2"
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      "t0"."id",
      "t0"."bigint_col" + 256 AS "i",
      "t0"."string_col" AS "s"
    FROM "functional_alltypes" AS "t0"
  ) AS "t1"
) AS "t3"
ORDER BY
  "t3"."id" ASC,
  "t3"."i" ASC,
  "t3"."s" ASC