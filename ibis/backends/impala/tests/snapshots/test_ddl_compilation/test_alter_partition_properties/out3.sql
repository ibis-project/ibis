ALTER TABLE tbl PARTITION (`year`=2007, `region`='CA')
SET TBLPROPERTIES (
  'bar'='2',
  'foo'='1'
)