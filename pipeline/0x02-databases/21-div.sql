-- Creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.
-- Aunction name: SafeDiv; arguments: a(INT), b(INT); Retunrs a/b

DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT)
    RETURNS FLOAT
    BEGIN
        SET @ans = 0;
        IF b <> 0 THEN
            set @ans = a/b;
        END IF;
        RETURN @ans;
    END //
DELIMITER ;
