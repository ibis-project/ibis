SELECT
  t2.region AS region,
  t2.total - t3.total AS diff
FROM (
  SELECT
    *
  FROM (
    SELECT
      t0.region AS region,
      t0.kind AS kind,
      SUM(t0.amount) AS total
    FROM purchases AS t0
    GROUP BY
      1,
      2
  ) AS t1
  WHERE
    (
      t1.kind = 'foo'
    )
) AS t2
INNER JOIN (
  SELECT
    *
  FROM (
    SELECT
      t0.region AS region,
      t0.kind AS kind,
      SUM(t0.amount) AS total
    FROM purchases AS t0
    GROUP BY
      1,
      2
  ) AS t1
  WHERE
    (
      t1.kind = 'bar'
    )
) AS t3
  ON t2.region = t3.region