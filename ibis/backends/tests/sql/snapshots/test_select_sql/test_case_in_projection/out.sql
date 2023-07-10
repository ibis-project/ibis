SELECT
  CASE t0.g WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS col1,
  CASE
    WHEN t0.g = 'foo'
    THEN 'bar'
    WHEN t0.g = 'baz'
    THEN t0.g
    ELSE CAST(NULL AS TEXT)
  END AS col2,
  t0.a AS a,
  t0.b AS b,
  t0.c AS c,
  t0.d AS d,
  t0.e AS e,
  t0.f AS f,
  t0.g AS g,
  t0.h AS h,
  t0.i AS i,
  t0.j AS j,
  t0.k AS k
FROM alltypes AS t0