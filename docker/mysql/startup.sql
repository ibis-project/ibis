CREATE USER 'ibis'@'localhost' IDENTIFIED BY 'ibis';
CREATE SCHEMA IF NOT EXISTS test_schema;
GRANT CREATE, DROP ON *.* TO 'ibis'@'%';
GRANT CREATE,SELECT,DROP ON `test_schema`.* TO 'ibis'@'%';
FLUSH PRIVILEGES;
