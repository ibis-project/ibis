WITH t0 AS (
  SELECT
    t1.person_id AS person_id,
    t1.birth_datetime AS birth_datetime,
    400 AS age
  FROM person AS t1
)
SELECT
  t0.person_id
FROM t0
WHERE
  t0.age <= 40