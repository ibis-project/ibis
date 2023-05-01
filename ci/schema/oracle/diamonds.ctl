options (SKIP=1)
load data
  infile '/opt/oracle/csv/diamonds.csv'
  into table "diamonds"
  fields terminated by "," optionally enclosed by '"'
  ( "carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z" )
