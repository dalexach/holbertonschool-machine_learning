-- Creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
-- Procedure: ComputeAverageScoreForUser

DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id_new INT)
    BEGIN
        UPDATE users SET average_score=(
            SELECT AVG(score) FROM corrections WHERE user_id=user_id_new)
            WHERE id=user_id_new;
    END//
DELIMITER ;
