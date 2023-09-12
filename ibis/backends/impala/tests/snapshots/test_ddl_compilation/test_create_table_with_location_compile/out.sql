CREATE TABLE `foo`.`another_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
STORED AS PARQUET
LOCATION '/path/to/table'