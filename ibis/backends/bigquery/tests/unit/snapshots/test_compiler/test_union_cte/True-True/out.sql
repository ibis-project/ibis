WITH t0 AS (
  SELECT
    t2.`string_col`,
    sum(t2.`double_col`) AS `metric`
  FROM functional_alltypes AS t2
  GROUP BY
    1
)
SELECT
  t1.`string_col`,
  t1.`metric`
FROM (
  WITH t0 AS (
    SELECT
      t2.`string_col`,
      sum(t2.`double_col`) AS `metric`
    FROM functional_alltypes AS t2
    GROUP BY
      1
  ), t2 AS (
    SELECT
      t3.`string_col`,
      t3.`metric`
    FROM (
      WITH t0 AS (
        SELECT
          t2.`string_col`,
          sum(t2.`double_col`) AS `metric`
        FROM functional_alltypes AS t2
        GROUP BY
          1
      )
      SELECT
        *
      FROM t0
      UNION DISTINCT
      SELECT
        t4.`string_col`,
        sum(t4.`double_col`) AS `metric`
      FROM functional_alltypes AS t4
      GROUP BY
        1
    ) AS t3
  )
  SELECT
    *
  FROM t2
  UNION DISTINCT
  SELECT
    t4.`string_col`,
    sum(t4.`double_col`) AS `metric`
  FROM functional_alltypes AS t4
  GROUP BY
    1
) AS t1