"""Tests for database functions with mocked psycopg2."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, call


class TestGetConnection:
    def test_success(self, app_module, mock_psycopg2):
        conn = app_module.get_connection("host", 5432, "db", "user", "pass")
        mock_psycopg2.assert_called_once()
        conn.set_session.assert_called_once_with(readonly=True, autocommit=True)

    def test_failure_propagates(self, app_module, mock_psycopg2):
        mock_psycopg2.side_effect = Exception("connection refused")
        with pytest.raises(Exception, match="connection refused"):
            app_module.get_connection("host", 5432, "db", "user", "pass")

    def test_connection_params(self, app_module, mock_psycopg2):
        app_module.get_connection("myhost", 5433, "mydb", "myuser", "mypass")
        kw = mock_psycopg2.call_args.kwargs
        assert kw["host"] == "myhost"
        assert kw["port"] == 5433
        assert kw["dbname"] == "mydb"
        assert kw["user"] == "myuser"
        assert kw["password"] == "mypass"


class TestFetchSchemas:
    def test_returns_list(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [("public",), ("sales",)]
        result = app_module.fetch_schemas("cid", "h", 5432, "d", "u", "p")
        assert result == ["public", "sales"]

    def test_empty(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = []
        result = app_module.fetch_schemas("cid", "h", 5432, "d", "u", "p")
        assert result == []

    def test_closes_resources(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [("public",)]
        app_module.fetch_schemas("cid", "h", 5432, "d", "u", "p")
        cursor.close.assert_called_once()
        mock_psycopg2.return_value.close.assert_called_once()


class TestFetchTables:
    def test_returns_tuples(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [("users", "BASE TABLE"), ("orders", "BASE TABLE")]
        result = app_module.fetch_tables("cid", "h", 5432, "d", "u", "p", "public")
        assert result == [("users", "BASE TABLE"), ("orders", "BASE TABLE")]

    def test_includes_views(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [("users", "BASE TABLE"), ("summary", "VIEW")]
        result = app_module.fetch_tables("cid", "h", 5432, "d", "u", "p", "public")
        assert len(result) == 2
        assert result[1][1] == "VIEW"

    def test_empty(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = []
        result = app_module.fetch_tables("cid", "h", 5432, "d", "u", "p", "public")
        assert result == []

    def test_uses_parameterized_query(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = []
        app_module.fetch_tables("cid", "h", 5432, "d", "u", "p", "public")
        call_args = cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "%s" in sql
        assert params == ("public",)


class TestFetchColumns:
    def test_returns_metadata(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            ("id", "integer", "NO", None),
            ("name", "varchar", "YES", None),
        ]
        result = app_module.fetch_columns("cid", "h", 5432, "d", "u", "p", "public", "users")
        assert len(result) == 2
        assert result[0][0] == "id"
        assert result[1][2] == "YES"

    def test_empty(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = []
        result = app_module.fetch_columns("cid", "h", 5432, "d", "u", "p", "public", "users")
        assert result == []


class TestFetchRowCount:
    def test_basic(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchone.return_value = (42,)
        result = app_module.fetch_row_count("cid", "h", 5432, "d", "u", "p", "public", "users")
        assert result == 42

    def test_zero(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchone.return_value = (0,)
        result = app_module.fetch_row_count("cid", "h", 5432, "d", "u", "p", "public", "users")
        assert result == 0

    def test_large(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchone.return_value = (1_000_000,)
        result = app_module.fetch_row_count("cid", "h", 5432, "d", "u", "p", "public", "users")
        assert result == 1_000_000


class TestRunQuery:
    def test_returns_dataframe(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = [{"id": 1, "name": "Alice"}]
        cursor.description = [("id",), ("name",)]
        df = app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_respects_limit(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = []
        cursor.description = [("id",)]
        app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1", limit=100)
        cursor.fetchmany.assert_called_with(100)

    def test_default_limit_500(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = []
        cursor.description = [("id",)]
        app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        cursor.fetchmany.assert_called_with(500)

    def test_empty_result(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = []
        cursor.description = [("id",)]
        df = app_module.run_query("h", 5432, "d", "u", "p", "SELECT 1")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_exception_propagates(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.execute.side_effect = Exception("syntax error")
        with pytest.raises(Exception, match="syntax error"):
            app_module.run_query("h", 5432, "d", "u", "p", "SELECT bad")


class TestGetSchemaSummary:
    def test_basic_format(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            ("public", "users", "id", "integer"),
            ("public", "users", "name", "varchar"),
        ]
        result = app_module.get_schema_summary("h", 5432, "d", "u", "p")
        assert "public.users:" in result
        assert "id (integer)" in result
        assert "name (varchar)" in result

    def test_multiple_tables(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            ("public", "users", "id", "integer"),
            ("public", "orders", "id", "integer"),
        ]
        result = app_module.get_schema_summary("h", 5432, "d", "u", "p")
        assert "public.users:" in result
        assert "public.orders:" in result
        # Tables separated by blank line
        assert "\n\n" in result

    def test_empty_schema(self, app_module, mock_psycopg2):
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchall.return_value = []
        result = app_module.get_schema_summary("h", 5432, "d", "u", "p")
        assert result == ""
