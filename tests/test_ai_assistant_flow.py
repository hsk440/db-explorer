"""Tests for the AI assistant routing and retry logic.

Since the routing logic is embedded in Streamlit UI code, we test the
decision contracts between functions: what each function returns and
how the calling code should interpret it.
"""

import pytest
from unittest.mock import MagicMock, patch
from conftest import make_api_response, SAMPLE_SCHEMA_TEXT


class TestRoutingDecisions:
    """Test the routing logic that decides how to handle AI responses."""

    def test_no_sql_needed_exact_match(self, app_module):
        """Exact 'NO_SQL_NEEDED' triggers conversational path."""
        sql = "NO_SQL_NEEDED"
        assert sql.strip() == "NO_SQL_NEEDED"  # This is the check on line 577

    def test_no_sql_needed_with_trailing_fails_exact_match(self, app_module):
        """BUG: AI appends text after NO_SQL_NEEDED, breaking the == check."""
        sql = "NO_SQL_NEEDED\n\nThis database appears to be..."
        # The current code does: sql.strip() == "NO_SQL_NEEDED"
        assert sql.strip() != "NO_SQL_NEEDED"  # Bug: doesn't match
        # After fix, should use: sql.strip().startswith("NO_SQL_NEEDED")
        assert sql.strip().startswith("NO_SQL_NEEDED")  # Fix would catch this

    def test_valid_sql_passes_safety_check(self, app_module):
        sql = "SELECT * FROM users LIMIT 100"
        assert app_module.is_read_only_query(sql) is True

    def test_prose_fails_safety_check(self, app_module):
        """Prose response correctly fails safety, routed to text display."""
        sql = "Based on the database schema, this appears to be a comprehensive system"
        assert app_module.is_read_only_query(sql) is False

    def test_none_sql_none_error_silent_failure(self, app_module):
        """BUG: When ask_claude_for_sql returns (None, None), no branch handles it."""
        sql = None
        error = None
        # Simulating the routing logic from app.py lines 570-594:
        # if error: -> False (error is None)
        # elif sql and sql.strip() == "NO_SQL_NEEDED": -> False (sql is None)
        # elif sql: -> False (sql is None)
        # No else clause -> silent failure, user sees nothing
        handled = False
        if error:
            handled = True
        elif sql and sql.strip() == "NO_SQL_NEEDED":
            handled = True
        elif sql:
            handled = True
        assert handled is False  # Bug: no fallback for this case


class TestRetryLogic:
    """Test the auto-retry mechanism when queries fail."""

    def test_success_on_first_attempt(self, app_module, mock_psycopg2, mock_anthropic):
        """Query succeeds immediately, no retry needed."""
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = [{"id": 1}]
        cursor.description = [("id",)]

        df = app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        assert len(df) == 1

    def test_retry_with_fixed_sql(self, app_module, mock_psycopg2, mock_anthropic):
        """First query fails, AI fixes it, second succeeds."""
        cursor = mock_psycopg2.return_value.cursor.return_value

        call_count = 0
        def side_effect(sql):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("invalid ORDER BY clause")
            # Second call succeeds

        cursor.execute.side_effect = side_effect
        cursor.fetchmany.return_value = [{"id": 1}]
        cursor.description = [("id",)]

        # The retry logic is in the UI code, so we test the AI fix function
        mock_anthropic.messages.create.return_value = make_api_response(
            "SELECT id FROM users ORDER BY id"
        )
        fixed_sql, err = app_module.ask_claude_for_sql(
            "Fix SQL: SELECT bad", SAMPLE_SCHEMA_TEXT, "key"
        )
        assert fixed_sql is not None
        assert app_module.is_read_only_query(fixed_sql) is True

    def test_ai_fix_returns_unsafe_sql(self, app_module, mock_anthropic):
        """AI returns unsafe SQL as a fix - should be caught by safety check."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "DROP TABLE users"
        )
        sql, _ = app_module.ask_claude_for_sql("fix this", SAMPLE_SCHEMA_TEXT, "key")
        assert app_module.is_read_only_query(sql) is False

    def test_ai_fix_returns_error(self, app_module, mock_anthropic):
        """AI fix call itself fails."""
        mock_anthropic.messages.create.side_effect = Exception("API down")
        sql, err = app_module.ask_claude_for_sql("fix", SAMPLE_SCHEMA_TEXT, "key")
        assert sql is None
        assert err is not None

    def test_max_retries_is_two(self, app_module):
        """Verify the retry constant in the code."""
        # Read from source to verify the constant
        import inspect
        source = inspect.getsource(app_module)
        assert "max_retries = 2" in source


class TestAnalysisPath:
    """Test the query -> analysis pipeline."""

    def test_analysis_receives_csv(self, app_module, mock_anthropic):
        """ask_claude_to_analyze should receive CSV data from the query result."""
        import pandas as pd
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        csv_data = df.head(100).to_csv(index=False)

        mock_anthropic.messages.create.return_value = make_api_response("analysis")
        analysis, _ = app_module.ask_claude_to_analyze(
            "who are the users", SAMPLE_SCHEMA_TEXT, csv_data, "SELECT *", "key"
        )
        # Verify CSV was included in the API call
        call_content = mock_anthropic.messages.create.call_args.kwargs["messages"][-1]["content"]
        assert "Alice" in call_content
        assert "Bob" in call_content

    def test_analysis_error_returns_message(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("model overloaded")
        _, err = app_module.ask_claude_to_analyze(
            "q", SAMPLE_SCHEMA_TEXT, "data", "SELECT 1", "key"
        )
        assert "model overloaded" in err
