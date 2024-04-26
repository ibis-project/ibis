SELECT
  "t5"."color"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t3"."color"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t1"."color"
        FROM (
          SELECT
            *
          FROM "t" AS "t0"
          WHERE
            LOWER("t0"."color") LIKE '%de%'
        ) AS "t1"
      ) AS "t2"
      WHERE
        CONTAINS(LOWER("t2"."color"), 'de')
    ) AS "t3"
  ) AS "t4"
  WHERE
    REGEXP_MATCHES(LOWER("t4"."color"), '.*ge.*')
) AS "t5"