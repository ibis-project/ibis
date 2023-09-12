CREATE EXTERNAL TABLE IF NOT EXISTS `foo`.`new_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
STORED AS PARQUET
LOCATION '/path/to/'