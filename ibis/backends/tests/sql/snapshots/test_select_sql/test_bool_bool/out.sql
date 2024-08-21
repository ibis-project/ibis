SELECT
  *
FROM "airlines" AS "t0"
WHERE
  (
    CAST("t0"."dest" AS BIGINT) = 0
  ) = TRUE