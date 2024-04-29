CREATE TABLE IF NOT EXISTS `tname`
STORED AS PARQUET
AS
SELECT
  *
FROM `functional_alltypes` AS `t0`
WHERE
  `t0`.`bigint_col` > 0