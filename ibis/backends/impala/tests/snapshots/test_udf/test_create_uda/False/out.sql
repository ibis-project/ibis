CREATE AGGREGATE FUNCTION `bar`.`test_name`(STRING, STRING) returns BIGINT location '/foo/bar.so'
init_fn='Init'
update_fn='Update'
merge_fn='Merge'
finalize_fn='Finalize'