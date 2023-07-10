SELECT
  t2.g AS g,
  SUM(t2.foo) AS "foo total"
FROM (
  SELECT
    t1.a AS a,
    t1.b AS b,
    t1.c AS c,
    t1.d AS d,
    t1.e AS e,
    t1.f AS f,
    t1.g AS g,
    t1.h AS h,
    t1.i AS i,
    t1.j AS j,
    t1.k AS k,
    t1.a + t1.b AS foo
  FROM (
    SELECT
      *
    FROM alltypes AS t0
    WHERE
      (
        t0.f > CAST(0 AS TINYINT)
      ) AND (
        (
          t0.a + t0.b
        ) < CAST(10 AS TINYINT)
      )
  ) AS t1
) AS t2
GROUP BY
  1
