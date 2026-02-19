ALTER TABLE tbl PARTITION (`year`=2007, `region`='CA')
SET SERDEPROPERTIES (
  'baz'='3'
)