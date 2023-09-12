CREATE EXTERNAL TABLE `foo`.`another_table`
STORED AS PARQUET
LOCATION '/path/to/table'
AS
SELECT t0.*
FROM `test1` t0