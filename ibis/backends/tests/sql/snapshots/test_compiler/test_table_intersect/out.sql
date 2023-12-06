SELECT
  t1.string_col AS key,
  CAST(t1.float_col AS DOUBLE) AS value
FROM (
  SELECT
    *
  FROM functional_alltypes AS t0
  WHERE
    (
      t0.int_col > CAST(0 AS TINYINT)
    )
) AS t1
INTERSECT
SELECT
  t2.string_col AS key,
  t2.double_col AS value
FROM (
  SELECT
    *
  FROM functional_alltypes AS t0
  WHERE
    (
      t0.int_col <= CAST(0 AS TINYINT)
    )
) AS t2