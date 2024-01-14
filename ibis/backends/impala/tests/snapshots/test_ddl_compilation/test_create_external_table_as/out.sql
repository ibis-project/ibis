CREATE EXTERNAL TABLE `foo`.`another_table`
STORED AS PARQUET
LOCATION '/path/to/table'
AS
SELECT
  *
FROM `test1`