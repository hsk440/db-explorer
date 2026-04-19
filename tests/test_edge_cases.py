"""Tests for edge cases: connection errors, API errors, fence stripping, data."""

import pytest
from unittest.mock import MagicMock
from conftest import make_api_response, SAMPLE_SCHEMA_TEXT
import pandas as pd


class TestConnectionEdgeCases:
    def test_connection_refused(self, app_module, mock_psycopg2):
        mock_psycopg2.side_effect = Exception("connection refused")
        with pytest.raises(Exception, match="connection refused"):
            app_module.get_connection("host", 5432, "db", "user", "pass")

    def test_dns_failure(self, app_module, mock_psycopg2):
        mock_psycopg2.side_effect = Exception("could not translate host name")
        with pytest.raises(Exception, match="could not translate"):
            app_module.get_connection("bad.host", 5432, "db", "user", "pass")

    def test_timeout(self, app_module, mock_psycopg2):
        mock_psycopg2.side_effect = Exception("timeout expired")
        with pytest.raises(Exception, match="timeout"):
            app_module.get_connection("host", 5432, "db", "user", "pass")


class TestApiEdgeCases:
    def test_rate_limited(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("rate limit exceeded")
        _, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert "rate limit" in err

    def test_auth_error(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("authentication error")
        _, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert "authentication" in err

    def test_api_timeout(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("request timed out")
        _, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert "timed out" in err

    def test_empty_content(self, app_module, mock_anthropic):
        """If the API returns empty text, we should not crash."""
        mock_anthropic.messages.create.return_value = make_api_response(text="")
        sql, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        # Either we get None+error, or an empty-string SQL — never a crash
        if sql is None:
            assert err is not None
        else:
            assert sql == ""

    def test_none_content(self, app_module, mock_anthropic):
        """If message.content is None (e.g. only tool_calls), we should not crash."""
        mock_anthropic.messages.create.return_value = make_api_response(text=None)
        sql, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        # Graceful handling — either error or empty
        if sql is None:
            assert err is not None
        else:
            assert sql == ""


class TestMarkdownFenceStripping:
    """Test the fence stripping logic in ask_claude_for_sql."""

    def test_lowercase_sql_fence(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "```sql\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_plain_fence(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "```\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_uppercase_sql_fence_fixed(self, app_module, mock_anthropic):
        """Fixed: uppercase ```SQL now stripped correctly."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "```SQL\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_postgresql_fence_fixed(self, app_module, mock_anthropic):
        """Fixed: ```postgresql tag now stripped correctly."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "```postgresql\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_no_fences(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("SELECT 1")
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_partial_fence_only_opening(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "```sql\nSELECT 1"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert "SELECT 1" in sql


class TestDataEdgeCases:
    def test_run_query_with_null_values(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = [{"id": 1, "name": None}]
        cursor.description = [("id",), ("name",)]
        df = app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        assert df.iloc[0]["name"] is None

    def test_run_query_large_result(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = [{"id": i} for i in range(500)]
        cursor.description = [("id",)]
        df = app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        assert len(df) == 500

    def test_csv_conversion(self, app_module):
        """DataFrame.to_csv should not error with mixed types."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", None],
            "amount": [99.99, 0.0],
        })
        csv = df.to_csv(index=False)
        assert "Alice" in csv
        assert "99.99" in csv
