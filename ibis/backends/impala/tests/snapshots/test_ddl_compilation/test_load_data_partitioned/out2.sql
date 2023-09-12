LOAD DATA INPATH '/path/to/data' OVERWRITE INTO TABLE `foo`.`functional_alltypes`
PARTITION (year=2007, month=7)