SELECT
  "t2"."person_id"
FROM (
  SELECT
    "t1"."person_id",
    "t1"."birth_datetime",
    "t1"."age"
  FROM (
    SELECT
      "t0"."person_id",
      "t0"."birth_datetime",
      CAST(400 AS SMALLINT) AS "age"
    FROM "person" AS "t0"
  ) AS "t1"
  WHERE
    "t1"."age" <= CAST(40 AS TINYINT)
) AS "t2"