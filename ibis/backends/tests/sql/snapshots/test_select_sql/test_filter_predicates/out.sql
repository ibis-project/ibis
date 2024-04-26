SELECT
  "t2"."color"
FROM (
  SELECT
    "t1"."color"
  FROM (
    SELECT
      "t0"."color"
    FROM "t" AS "t0"
    WHERE
      LOWER("t0"."color") LIKE '%de%'
  ) AS "t1"
  WHERE
    CONTAINS(LOWER("t1"."color"), 'de')
) AS "t2"
WHERE
  REGEXP_MATCHES(LOWER("t2"."color"), '.*ge.*')