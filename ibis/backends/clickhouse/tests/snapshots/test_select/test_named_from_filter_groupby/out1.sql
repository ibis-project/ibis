SELECT
  t1.key AS key,
  SUM((
    (
      t1.value + 1
    ) + 2
  ) + 3) AS abc
FROM (
  SELECT
    t0.key AS key,
    t0.value AS value
  FROM t0 AS t0
  WHERE
    t0.value = 42
) AS t1
GROUP BY
  t1.key