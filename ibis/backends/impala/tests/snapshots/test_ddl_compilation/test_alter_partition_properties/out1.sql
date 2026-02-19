ALTER TABLE tbl PARTITION (`year`=2007, `region`='CA')
SET LOCATION '/users/foo/my-data'