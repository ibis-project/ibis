CREATE TABLE `foo`.`another_table`
(`foo` STRING,
 `bar` TINYINT,
 `baz` SMALLINT)
STORED AS PARQUET
LOCATION '/path/to/table'