"""Tests for pure functions: is_read_only_query, conn_id, conn_params."""

import pytest


# ===================================================================
# is_read_only_query
# ===================================================================

class TestIsReadOnlyQuery:
    """Tests for the SQL safety check function."""

    # --- Safe queries (should return True) ---

    def test_select_simple(self, app_module):
        assert app_module.is_read_only_query("SELECT * FROM users") is True

    def test_select_lowercase(self, app_module):
        assert app_module.is_read_only_query("select id from orders") is True

    def test_select_mixed_case(self, app_module):
        assert app_module.is_read_only_query("SeLeCt 1") is True

    def test_with_cte(self, app_module):
        assert app_module.is_read_only_query(
            "WITH cte AS (SELECT 1) SELECT * FROM cte"
        ) is True

    def test_explain(self, app_module):
        assert app_module.is_read_only_query("EXPLAIN SELECT * FROM users") is True

    def test_show(self, app_module):
        assert app_module.is_read_only_query("SHOW server_version") is True

    def test_values(self, app_module):
        assert app_module.is_read_only_query("VALUES (1, 'a'), (2, 'b')") is True

    def test_leading_whitespace(self, app_module):
        assert app_module.is_read_only_query("   \n  SELECT 1") is True

    def test_line_comment_before_select(self, app_module):
        assert app_module.is_read_only_query("-- comment\nSELECT 1") is True

    def test_block_comment_before_select(self, app_module):
        assert app_module.is_read_only_query("/* block */SELECT 1") is True

    def test_multiple_comments_before_select(self, app_module):
        sql = "-- line comment\n/* block */ SELECT 1"
        assert app_module.is_read_only_query(sql) is True

    def test_select_with_subquery(self, app_module):
        sql = "SELECT * FROM (SELECT id FROM users) sub"
        assert app_module.is_read_only_query(sql) is True

    def test_select_with_update_in_column_name(self, app_module):
        """Column names containing dangerous keywords should be safe."""
        sql = "SELECT updated_at, created_by, call_duration FROM records"
        assert app_module.is_read_only_query(sql) is True

    def test_select_with_union(self, app_module):
        sql = "SELECT 1 UNION ALL SELECT 2"
        assert app_module.is_read_only_query(sql) is True

    # --- Blocked queries (should return False) ---

    def test_insert_blocked(self, app_module):
        assert app_module.is_read_only_query("INSERT INTO users VALUES (1)") is False

    def test_update_blocked(self, app_module):
        assert app_module.is_read_only_query("UPDATE users SET name='x'") is False

    def test_delete_blocked(self, app_module):
        assert app_module.is_read_only_query("DELETE FROM users") is False

    def test_drop_blocked(self, app_module):
        assert app_module.is_read_only_query("DROP TABLE users") is False

    def test_create_blocked(self, app_module):
        assert app_module.is_read_only_query("CREATE TABLE foo (id int)") is False

    def test_truncate_blocked(self, app_module):
        assert app_module.is_read_only_query("TRUNCATE TABLE users") is False

    def test_alter_blocked(self, app_module):
        assert app_module.is_read_only_query("ALTER TABLE users ADD col int") is False

    def test_empty_string(self, app_module):
        assert app_module.is_read_only_query("") is False

    def test_whitespace_only(self, app_module):
        assert app_module.is_read_only_query("   \n  ") is False

    def test_comment_only(self, app_module):
        assert app_module.is_read_only_query("-- just a comment") is False

    def test_block_comment_only(self, app_module):
        assert app_module.is_read_only_query("/* just a block comment */") is False

    # --- Known bug scenarios ---

    def test_no_sql_needed_literal(self, app_module):
        """NO_SQL_NEEDED doesn't start with a safe keyword."""
        assert app_module.is_read_only_query("NO_SQL_NEEDED") is False

    def test_prose_response(self, app_module):
        """AI returning prose instead of SQL (known bug from logs)."""
        assert app_module.is_read_only_query(
            "Based on the database schema, this appears to be a comprehensive"
        ) is False

    def test_no_sql_needed_with_trailing_text(self, app_module):
        """AI returning NO_SQL_NEEDED with explanation appended."""
        assert app_module.is_read_only_query(
            "NO_SQL_NEEDED\n\nThis database appears to be..."
        ) is False

    # --- Comment bypass attempts ---

    def test_comment_hiding_insert(self, app_module):
        sql = "-- SELECT\nINSERT INTO users VALUES (1)"
        assert app_module.is_read_only_query(sql) is False

    def test_block_comment_hiding_insert(self, app_module):
        sql = "/* SELECT */ INSERT INTO users VALUES (1)"
        assert app_module.is_read_only_query(sql) is False


# ===================================================================
# conn_id
# ===================================================================

class TestConnId:
    def test_format(self, app_module, session_state):
        session_state["host"] = "db.example.com"
        session_state["port"] = "5432"
        session_state["dbname"] = "mydb"
        session_state["user"] = "admin"
        assert app_module.conn_id() == "db.example.com:5432/mydb/admin"

    def test_with_ip(self, app_module, session_state):
        session_state["host"] = "192.168.1.1"
        session_state["port"] = "5433"
        session_state["dbname"] = "test-db"
        session_state["user"] = "my_user"
        assert app_module.conn_id() == "192.168.1.1:5433/test-db/my_user"


# ===================================================================
# conn_params
# ===================================================================

class TestConnParams:
    def test_basic(self, app_module, session_state):
        session_state["host"] = "localhost"
        session_state["port"] = "5432"
        session_state["dbname"] = "test"
        session_state["user"] = "admin"
        session_state["password"] = "pass"
        result = app_module.conn_params()
        assert result == {
            "host": "localhost",
            "port": 5432,
            "dbname": "test",
            "user": "admin",
            "password": "pass",
        }

    def test_port_conversion(self, app_module, session_state):
        session_state["port"] = "9999"
        result = app_module.conn_params()
        assert result["port"] == 9999
        assert isinstance(result["port"], int)

    def test_invalid_port_raises(self, app_module, session_state):
        session_state["port"] = "abc"
        with pytest.raises(ValueError):
            app_module.conn_params()

    def test_empty_port_raises(self, app_module, session_state):
        session_state["port"] = ""
        with pytest.raises(ValueError):
            app_module.conn_params()
