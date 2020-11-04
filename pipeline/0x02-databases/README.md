# [holbertonschool-machine_learning](https://github.com/dalexach/holbertonschool-machine_learning)

## 0x02. Databases
### Description 
After fetching data via APIs, storing them is also really important for training a Machine Learning model.

You have multiple option:

- Relation database
- Not Relation database
- Key-Value storage
- Document storage
- Data Lake
- etc.

In this project, you will touch the first 2: relation and not relation database.

Relation databases are mainly used for application, not for source of data for training your ML models, but it can be really useful for the data processing, labeling and injection in another data storage. In this project, you will play with basic SQL commands but also create automation and computing on your data directly in SQL - less load at your application level since the computing power is dispatched to the database.

Not relation databases, known as NoSQL, will give you flexibility on your data: document, versioning, not a fix schema, no validation to improve performance, complex lookup, etc.

### Files
#### Mandatory Tasks

| File | Description |
| ------ | ------ |
| [0-create_database_if_missing.sql](0-create_database_if_missing.sql) | Script that creates the database db_0 in your MySQL server. |
| [1-first_table.sql](1-first_table.sql) | Script that creates a table called first_table in the current database in your MySQL server. |
| [2-list_values.sql](2-list_values.sql) | Script that lists all rows of the table first_table in your MySQL server. |
| [3-insert_value.sql](3-insert_value.sql) | Script that inserts a new row in the table first_table in your MySQL server. |
| [4-best_score.sql](4-best_score.sql) | Script that lists all records with a score >= 10 in the table second_table in your MySQL server. |
| [5-average.sql](5-average.sql) | Script that computes the score average of all records in the table second_table in your MySQL server. |
| [6-avg_temperatures.sql](6-avg_temperatures.sql) | Script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending). |
| [7-max_state.sql](7-max_state.sql) | Script that displays the max temperature of each state (ordered by State name). |
| [8-genre_id_by_show.sql](8-genre_id_by_show.sql) | Script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked. |
| [9-no_genre.sql](9-no_genre.sql) | Script that lists all shows contained in hbtn_0d_tvshows without a genre linked. |
| [10-count_shows_by_genre.sql](10-count_shows_by_genre.sql) | Script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each. |
| [11-rating_shows.sql](11-rating_shows.sql) | Script that lists all shows from hbtn_0d_tvshows_rate by their rating. |
| [12-rating_genres.sql](12-rating_genres.sql) | Script that lists all genres in the database hbtn_0d_tvshows_rate by their rating. |
| [13-uniq_users.sql](13-uniq_users.sql) | Script that creates a table users with these attributes:<li> id, integer, never null, auto increment and primary key</li><li>email, string (255 characters), never null and unique</li> <li>name, string (255 characters)</li> |
| [14-country_users.sql](14-country_users.sql) | Script that creates a table users with these attributes:<li>id, integer, never null, auto increment and primary key</li> <li>email, string (255 characters), never null and unique</li> <li> name, string (255 characters)</li> <li> country, enumeration of countries: US, CO and TN, never null (= default will be the firs element of the enumeration, here US)</li>. |
| [15-fans.sql](15-fans.sql) | Script that ranks country origins of bands, ordered by the number of (non-unique) fans. |
| [16-glam_rock.sql](16-glam_rock.sql) | Script that lists all bands with Glam as their main style, ranked by their longevity. |
| [17-store.sql](17-store.sql) | Script that creates a trigger that decreases the quantity of an item after adding a new order. |
| [18-valid_email.sql](18-valid_email.sql) | Script that creates a trigger that resets the attribute valid_email only when the email has been changed. |
| [19-bonus.sql](19-bonus.sql) | Script that creates a stored procedure AddBonus that adds a new correction for a student. |
| [20-average_score.sql](20-average_score.sql) | Script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student. |
| [21-div.sql](21-div.sql) | Script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0. |
| [22-list_databases](22-list_databases) | Script that lists all databases in MongoDB. |
| [23-use_or_create_database](23-use_or_create_database) | Script that creates or uses the database my_db. |
| [24-insert](24-insert) | Script that inserts a document in the collection school. |
| [25-all](25-all) | Script that lists all documents in the collection school. |
| [26-match](26-match) | Script that lists all documents with name="Holberton school" in the collection school. |
| [27-count](27-count) | Script that displays the number of documents in the collection school. |
| [28-update](28-update) | Script that adds a new attribute to a document in the collection school. |
| [29-delete](29-delete) | Script that deletes all documents with name="Holberton school" in the collection school. |
| [30-all.py](30-all.py) | Python function that lists all documents in a collection. |
| [31-insert_school.py](31-insert_school.py) | Python function that inserts a new document in a collection based on kwargs. |
| [32-update_topics.py](32-update_topics.py) | Python function that changes all topics of a school document based on the name. |
| [33-schools_by_topic.py](33-schools_by_topic.py) | Python function that returns the list of school having a specific topic. |
| [34-log_stats.py](34-log_stats.py) | Python script that provides some stats about Nginx logs stored in MongoDB. |

#### Advanced Tasks

| File | Description |
| ------ | ------ |
| [100-index_my_names.sql](100-index_my_names.sql) |  SQL script that creates an index idx_name_first on the table names and the first letter of name. |
| [101-index_name_score.sql](101-index_name_score.sql) | SQL script that creates an index idx_name_first_score on the table names and the first letter of name and the score. |
| [102-need_meeting.sql](102-need_meeting.sql) | SQL script that creates a view need_meeting that lists all students that have a score under 80 (strict) and no last_meeting or more than 1 month. |
| [103-average_weighted_score.sql](103-average_weighted_score.sql) | SQL script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student. |
| [104-find](104-find) | Script that lists all documents with name starting by Holberton in the collection school |
| [105-students.py](105-students.py) | Python function that returns all students sorted by average score. |
| [106-log_stats.py](106-log_stats.py) | Improve 34-log_stats.py by adding the top 10 of the most present IPs in the collection nginx of the database logs. |
<!----
| []() | . |
--->

### Build with
- Python (python 3.5)
- Ubuntu 16.04 LTS 
- MySQL (mysql 5.7.30)
- MongoDB (mongodb 4.2)
- PyMongo (pymongo 3.10)

## Author

[Daniela Chamorro](https://www.linkedin.com/in/dalexach/) [:octocat:](https://github.com/dalexach)

[Twitter](https://twitter.com/dalexach)
