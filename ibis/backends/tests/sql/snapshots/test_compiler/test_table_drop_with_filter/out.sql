SELECT
  *
FROM (
  SELECT
    t3.a AS a
  FROM (
    SELECT
      t2.a AS a,
      t2.b AS b,
      MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0) AS the_date
    FROM (
      SELECT
        *
      FROM t AS t1
      WHERE
        (
          t1.c = MAKE_TIMESTAMP(2018, 1, 1, 0, 0, 0.0)
        )
    ) AS t2
  ) AS t3
  INNER JOIN s AS t0
    ON t3.b = t0.b
) AS t5
WHERE
  (
    t5.a < CAST(1.0 AS DOUBLE)
  )