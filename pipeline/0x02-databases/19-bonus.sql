-- Creates a stored procedure AddBonus that adds a new correction for a student
-- Procedure AddBonus

DELIMITER //
CREATE PROCEDURE AddBonus(
    IN user_id_new INT,
    IN project_name VARCHAR(255),
    IN score_new INT)
    BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
            INSERT INTO projects(name) VALUES (project_name);
        END IF;
        INSERT INTO corrections(user_id, project_id, score)
            VALUES(user_id_new,
                  (SELECT id FROM projects WHERE name=project_name),
                  score_new);
    END //
DELIMITER ;
