CREATE TABLE `bar`.`some_table`
STORED AS PARQUET
AS
SELECT
  *
FROM `functional_alltypes` AS `t0`
WHERE
  `t0`.`bigint_col` > 0