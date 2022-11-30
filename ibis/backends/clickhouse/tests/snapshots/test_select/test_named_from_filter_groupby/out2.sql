SELECT
  t0.key,
  SUM((
    (
      t0.value + 1
    ) + 2
  ) + 3) AS foo
FROM t0 AS t0
WHERE
  t0.value = 42
GROUP BY
  t0.key