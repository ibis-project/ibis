SELECT
  approx_quantiles(IF(`t0`.`month` > 0, `t0`.`double_col`, NULL), IF(`t0`.`month` > 0, 2, NULL))[offset(1)] AS `ApproxMedian_double_col_ Greater_month_ 0`
FROM `functional_alltypes` AS `t0`