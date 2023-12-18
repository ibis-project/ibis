SELECT
  t4.date AS date,
  t4.value AS value
FROM (
  SELECT
    t0.date AS date,
    t0.A AS value
  FROM X1 AS t0
  UNION ALL
  SELECT
    t1.date AS date,
    t1.B AS value
  FROM X2 AS t1
) AS t4
WHERE
  t4.date >= '2023-01-01' AND NOT t4.value IS NULL