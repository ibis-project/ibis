ALTER TABLE tbl PARTITION (year=2007, month=4)
SET TBLPROPERTIES (
  'bar'='2',
  'foo'='1'
)