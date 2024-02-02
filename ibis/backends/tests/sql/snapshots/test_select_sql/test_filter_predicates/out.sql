SELECT
  "t0"."color"
FROM "t" AS "t0"
WHERE
  LOWER("t0"."color") LIKE '%de%'
  AND CONTAINS(LOWER("t0"."color"), 'de')
  AND REGEXP_MATCHES(LOWER("t0"."color"), '.*ge.*')