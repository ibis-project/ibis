SELECT
  t1.person_id AS person_id
FROM (
  SELECT
    *
  FROM person AS t0
  WHERE
    (
      CAST(400 AS SMALLINT) <= CAST(40 AS TINYINT)
    )
) AS t1