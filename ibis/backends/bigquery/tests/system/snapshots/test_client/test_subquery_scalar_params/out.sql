WITH t0 AS (
  SELECT
    t2.`float_col`,
    t2.`timestamp_col`,
    t2.`int_col`,
    t2.`string_col`
  FROM `ibis-gbq`.ibis_gbq_testing.functional_alltypes AS t2
  WHERE
    t2.`timestamp_col` < @param_0
)
SELECT
  count(t1.`foo`) AS `count`
FROM (
  SELECT
    t0.`string_col`,
    sum(t0.`float_col`) AS `foo`
  FROM t0
  GROUP BY
    1
) AS t1