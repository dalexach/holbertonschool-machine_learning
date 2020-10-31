-- Creates a table with requirements
-- table: users; atributes: id=INT, email=STRING(255), name=STRING(255); database: holberton

CREATE TABLE IF NOT EXISTS users(id INT NOT NULL AUTO_INCREMENT, email VARCHAR(255) NOT NULL UNIQUE, name VARCHAR(255), PRIMARY KEY(id));
