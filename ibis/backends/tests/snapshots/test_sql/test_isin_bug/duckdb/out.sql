SELECT
  t0.x IN (
    SELECT
      t1.x
    FROM (
      SELECT
        t0.x AS x
      FROM t AS t0
      WHERE
        t0.x > CAST(2 AS TINYINT)
    ) AS t1
  ) AS "InColumn(x, x)"
FROM t AS t0