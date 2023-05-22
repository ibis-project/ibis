CREATE EXTERNAL TABLE IF NOT EXISTS `foo`.`new_table`
LIKE PARQUET '/path/to/parquetfile'
STORED AS PARQUET
LOCATION '/path/to/'