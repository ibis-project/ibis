-- SingleStoreDB initialization script for Ibis testing

-- Create the testing database
CREATE DATABASE IF NOT EXISTS ibis_testing;

-- Create a test user with appropriate permissions
CREATE USER IF NOT EXISTS 'ibis'@'%' IDENTIFIED BY 'ibis';
GRANT ALL PRIVILEGES ON ibis_testing.* TO 'ibis'@'%';

-- Create some basic test tables for validation
CREATE TABLE IF NOT EXISTS ibis_testing.simple_table (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    value DECIMAL(10,2)
);

-- Insert some test data
INSERT IGNORE INTO ibis_testing.simple_table VALUES
    (1, 'test1', 100.50),
    (2, 'test2', 200.75),
    (3, 'test3', 300.25);

-- Create a table demonstrating SingleStoreDB-specific types
CREATE TABLE IF NOT EXISTS ibis_testing.singlestore_types (
    id INT PRIMARY KEY AUTO_INCREMENT,
    json_data JSON,
    binary_data BLOB,
    geom_data GEOGRAPHYPOINT,
    timestamp_col TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert test data for SingleStoreDB types
INSERT IGNORE INTO ibis_testing.singlestore_types (json_data, binary_data, geom_data) VALUES
    ('{"key": "value1", "number": 123}', UNHEX('48656C6C6F'), 'POINT(1 1)'),
    ('{"key": "value2", "array": [1,2,3]}', UNHEX('576F726C64'), 'POINT(2 2)');

-- Show that the initialization completed
SELECT 'SingleStoreDB initialization completed successfully' AS status;
