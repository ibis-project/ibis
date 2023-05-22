CREATE TABLE IF NOT EXISTS `tname`
STORED AS PARQUET
AS
SELECT t0.*
FROM `functional_alltypes` t0
WHERE t0.`bigint_col` > 0