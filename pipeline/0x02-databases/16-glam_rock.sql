-- Lists all bands with Glam as their main style, ranked by their longevity
-- database: holberton; table: metal_bands

SELECT band_name, IF(split IS NULL, (2020 - formed), split - formed) AS lifespan FROM metal_bands WHERE style LIKE '%Glam Rock%' ORDER BY lifespan DESC;
