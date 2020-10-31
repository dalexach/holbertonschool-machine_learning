-- Llists all records with a condition in the table in your MySQL server
-- condition: score>=10; table: second_table

SELECT score, name FROM second_table WHERE score>=10 ORDER BY score DESC;
