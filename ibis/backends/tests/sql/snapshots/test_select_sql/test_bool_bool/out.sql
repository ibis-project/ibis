SELECT
  "t0"."dest",
  "t0"."origin",
  "t0"."arrdelay"
FROM "airlines" AS "t0"
WHERE
  (
    CAST("t0"."dest" AS BIGINT) = CAST(0 AS TINYINT)
  ) = TRUE