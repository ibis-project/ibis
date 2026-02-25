SELECT
  APPROX_QUANTILES(IF(`t0`.`string_col` > 'a', `t0`.`float_col`, NULL), 2 IGNORE NULLS)[1] AS `ApproxMedian_float_col_Greater_string_col_'a'`
FROM `functional_alltypes` AS `t0`