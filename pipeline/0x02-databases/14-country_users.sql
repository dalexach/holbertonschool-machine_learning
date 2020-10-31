-- Create a table with requirements
-- table: users; attributes: id(int), email(string), name(string), country(enumerate)

CREATE TABLE IF NOT EXISTS users(id INT NOT NULL AUTO_INCREMENT, email VARCHAR(255) NOT NULL UNIQUE, name VARCHAR(255), country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US', PRIMARY KEY(id));
