SELECT
  t0.person_id
FROM (
  SELECT
    t1.person_id AS person_id,
    t1.birth_datetime AS birth_datetime,
    400 AS age
  FROM person AS t1
) AS t0
WHERE
  t0.age <= 40