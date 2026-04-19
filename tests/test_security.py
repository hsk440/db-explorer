"""Security tests: SQL injection, read-only enforcement."""

import pytest
from unittest.mock import patch, MagicMock, call


class TestReadOnlyEnforcement:
    """Verify the read-only chain: client-side check + connection-level."""

    def test_connection_options_readonly(self, app_module, mock_psycopg2):
        """psycopg2.connect must be called with read-only transaction option."""
        app_module.get_connection("host", 5432, "db", "user", "pass")
        connect_call = mock_psycopg2.call_args
        assert connect_call.kwargs["options"] == "-c default_transaction_read_only=on"

    def test_connection_set_session_readonly(self, app_module, mock_psycopg2):
        """Connection must have readonly=True set on the session."""
        conn = app_module.get_connection("host", 5432, "db", "user", "pass")
        conn.set_session.assert_called_once_with(readonly=True, autocommit=True)

    def test_connection_timeout(self, app_module, mock_psycopg2):
        """Connection must use a 10-second timeout."""
        app_module.get_connection("host", 5432, "db", "user", "pass")
        connect_call = mock_psycopg2.call_args
        assert connect_call.kwargs["connect_timeout"] == 10


class TestSqlInjectionAttempts:
    """Test that various injection patterns are handled."""

    def test_comment_bypass(self, app_module):
        assert app_module.is_read_only_query("/**/INSERT INTO users VALUES(1)") is False

    def test_semicolon_append(self, app_module):
        """Semicolon-appended commands pass client check but DB blocks them.
        This documents a known limitation - the DB read-only mode is the real safety net."""
        result = app_module.is_read_only_query("SELECT 1; DROP TABLE users")
        # Client check only validates the START keyword - this passes
        assert result is True  # DB-level protection is the real guard

    def test_select_into(self, app_module):
        """SELECT INTO passes client check but fails at DB level (read-only)."""
        result = app_module.is_read_only_query("SELECT * INTO new_table FROM users")
        assert result is True  # DB-level protection handles this

    def test_copy_blocked(self, app_module):
        assert app_module.is_read_only_query("COPY users TO '/tmp/dump'") is False

    def test_grant_blocked(self, app_module):
        assert app_module.is_read_only_query("GRANT ALL ON users TO attacker") is False

    def test_multiline_comment_bypass(self, app_module):
        sql = "/*\nSELECT 1\n*/\nINSERT INTO users VALUES(1)"
        assert app_module.is_read_only_query(sql) is False

    def test_unicode_bypass(self, app_module):
        """Fullwidth characters should not match safe keywords."""
        assert app_module.is_read_only_query("\uff33\uff25\uff2c\uff25\uff23\uff34 1") is False


class TestFetchRowCountSqlFormat:
    """fetch_row_count uses f-string for schema/table - document the risk."""

    def test_normal_table(self, app_module, mock_psycopg2, mock_cursor):
        mock_cursor = mock_psycopg2.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = (42,)
        count = app_module.fetch_row_count("cid", "host", 5432, "db", "user", "pass", "public", "users")
        # Verify the SQL uses double-quoted identifiers
        sql_called = mock_cursor.execute.call_args[0][0]
        assert '"public"."users"' in sql_called

    def test_special_chars_in_table_name(self, app_module, mock_psycopg2, mock_cursor):
        """Table name with quotes could break the f-string SQL.
        Documents this weakness - data comes from information_schema (safe) but
        the pattern is fragile."""
        mock_cursor = mock_psycopg2.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = (0,)
        # A table name with embedded double-quote
        app_module.fetch_row_count("cid", "host", 5432, "db", "user", "pass", "public", 'user"s')
        sql_called = mock_cursor.execute.call_args[0][0]
        # The embedded " breaks the quoting: "user"s" -> SQL error
        assert 'user"s' in sql_called
