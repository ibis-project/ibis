CREATE TABLE IF NOT EXISTS `tname`
STORED AS PARQUET
AS
SELECT *
FROM functional_alltypes
WHERE `bigint_col` > 0