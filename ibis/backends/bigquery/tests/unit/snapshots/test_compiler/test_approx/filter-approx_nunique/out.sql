SELECT
  APPROX_COUNT_DISTINCT(IF(t0.`month` > 0, t0.`double_col`, NULL)) AS `ApproxCountDistinct_double_col_ Greater_month_ 0`
FROM functional_alltypes AS t0