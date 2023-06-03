CREATE USER 'ibis'@'localhost' IDENTIFIED BY 'ibis';
GRANT CREATE, DROP ON *.* TO 'ibis'@'%';
FLUSH PRIVILEGES;
