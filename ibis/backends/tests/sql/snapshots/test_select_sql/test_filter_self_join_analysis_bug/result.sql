SELECT
  t4.region,
  t4.total - t5.total AS diff
FROM (
  SELECT
    t1.region,
    t1.kind,
    t1.total
  FROM (
    SELECT
      t0.region,
      t0.kind,
      SUM(t0.amount) AS total
    FROM purchases AS t0
    GROUP BY
      1,
      2
  ) AS t1
  WHERE
    t1.kind = 'foo'
) AS t4
INNER JOIN (
  SELECT
    t1.region,
    t1.kind,
    t1.total
  FROM (
    SELECT
      t0.region,
      t0.kind,
      SUM(t0.amount) AS total
    FROM purchases AS t0
    GROUP BY
      1,
      2
  ) AS t1
  WHERE
    t1.kind = 'bar'
) AS t5
  ON t4.region = t5.region