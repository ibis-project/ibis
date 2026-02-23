ALTER TABLE tbl ADD PARTITION (`year`=2007, `region`='CA')
LOCATION '/users/foo/my-data'