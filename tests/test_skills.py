"""Tests for the skills module: tool definitions, dispatchers, and artifact builders."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

import skills


# ===================================================================
# Tool definitions
# ===================================================================

class TestToolDefinitions:
    def test_three_tools_defined(self):
        names = [t["name"] for t in skills.TOOL_DEFINITIONS]
        assert "query_database" in names
        assert "create_excel_artifact" in names
        assert "create_word_artifact" in names

    def test_tools_have_schemas(self):
        for tool in skills.TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]
            assert "required" in tool["input_schema"]

    def test_query_database_requires_query(self):
        tool = [t for t in skills.TOOL_DEFINITIONS if t["name"] == "query_database"][0]
        assert "query" in tool["input_schema"]["required"]

    def test_excel_requires_filename_and_sheets(self):
        tool = [t for t in skills.TOOL_DEFINITIONS if t["name"] == "create_excel_artifact"][0]
        assert "filename" in tool["input_schema"]["required"]
        assert "sheets" in tool["input_schema"]["required"]


# ===================================================================
# Result truncation (token optimization)
# ===================================================================

class TestTruncateQueryResult:
    def test_empty_df(self):
        result = skills.truncate_query_result(pd.DataFrame())
        assert result["status"] == "ok"
        assert result["row_count"] == 0
        assert result["rows"] == []

    def test_small_result_kept_full(self):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        result = skills.truncate_query_result(df)
        assert result["row_count"] == 3
        assert len(result["rows"]) == 3

    def test_medium_result_truncated(self):
        """51+ rows get split into head + tail + summary."""
        df = pd.DataFrame({"id": range(100), "val": [f"v{i}" for i in range(100)]})
        result = skills.truncate_query_result(df)
        assert result["row_count"] == 100
        assert "rows_head" in result
        assert "rows_tail" in result
        assert "summary" in result
        assert len(result["rows_head"]) == 20
        assert len(result["rows_tail"]) == 5

    def test_null_values_become_none(self):
        df = pd.DataFrame({"id": [1], "name": [None]})
        result = skills.truncate_query_result(df)
        assert result["rows"][0]["name"] is None


# ===================================================================
# Excel artifact builder
# ===================================================================

class TestBuildExcelArtifact:
    def test_single_sheet(self):
        art = skills.build_excel_artifact(
            filename="test.xlsx",
            title="Test",
            summary="Test summary",
            sheets={"Data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
        )
        assert art["type"] == "excel"
        assert art["filename"] == "test.xlsx"
        assert art["title"] == "Test"
        assert art["size_bytes"] > 0
        assert isinstance(art["bytes"], bytes)
        assert art["sheet_count"] == 1
        # xlsx files start with PK (zip signature)
        assert art["bytes"][:2] == b"PK"

    def test_multiple_sheets(self):
        art = skills.build_excel_artifact(
            filename="multi.xlsx",
            title="Multi",
            summary="",
            sheets={
                "Billing": [{"table": "invoices"}],
                "Support": [{"table": "tickets"}],
                "Network": [{"table": "switches"}],
            },
        )
        assert art["sheet_count"] == 3
        assert art["bytes"][:2] == b"PK"

    def test_auto_extension(self):
        art = skills.build_excel_artifact(
            filename="no_ext",
            title="T",
            summary="",
            sheets={"S": [{"a": 1}]},
        )
        assert art["filename"] == "no_ext.xlsx"

    def test_empty_sheet_gets_placeholder(self):
        """Empty sheet list should produce a valid file, not crash."""
        art = skills.build_excel_artifact(
            filename="empty.xlsx",
            title="T",
            summary="",
            sheets={"Empty": []},
        )
        assert art["size_bytes"] > 0


# ===================================================================
# Word artifact builder
# ===================================================================

class TestBuildWordArtifact:
    def test_basic_report(self):
        art = skills.build_word_artifact(
            filename="report.docx",
            title="Report",
            summary="Test",
            sections=[
                {"heading": "Intro", "body": "This is the intro."},
                {"heading": "Findings", "body": "Key findings here."},
            ],
        )
        assert art["type"] == "word"
        assert art["filename"] == "report.docx"
        assert art["section_count"] == 2
        assert art["bytes"][:2] == b"PK"  # docx is also a zip

    def test_section_with_table(self):
        art = skills.build_word_artifact(
            filename="tabled.docx",
            title="With Table",
            summary="",
            sections=[
                {
                    "heading": "Data",
                    "body": "See below:",
                    "table_data": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                }
            ],
        )
        assert art["size_bytes"] > 0

    def test_markdown_in_body(self):
        """Should not crash on bullets, bold, headings."""
        art = skills.build_word_artifact(
            filename="md.docx",
            title="MD",
            summary="",
            sections=[
                {
                    "heading": "Test",
                    "body": "Intro.\n\n## Subheading\n\n- bullet **bold**\n- another",
                }
            ],
        )
        assert art["size_bytes"] > 0


# ===================================================================
# Tool dispatcher
# ===================================================================

class TestDispatchTool:
    def test_query_database_success(self):
        def mock_run_query(**kwargs):
            return pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})

        def mock_is_safe(sql):
            return True

        result, artifact = skills.dispatch_tool(
            "query_database",
            {"query": "SELECT * FROM t", "purpose": "test"},
            mock_run_query, mock_is_safe, {"host": "h"},
        )
        assert result["status"] == "ok"
        assert result["row_count"] == 2
        assert artifact is None

    def test_query_database_blocks_unsafe(self):
        result, artifact = skills.dispatch_tool(
            "query_database",
            {"query": "DROP TABLE users", "purpose": "evil"},
            lambda **k: pd.DataFrame(),
            lambda sql: False,  # unsafe
            {},
        )
        assert result["status"] == "error"
        assert "read-only" in result["error"].lower() or "select" in result["error"].lower()

    def test_query_database_returns_error_on_failure(self):
        def failing_query(**kwargs):
            raise Exception("column ambiguous")

        result, artifact = skills.dispatch_tool(
            "query_database",
            {"query": "SELECT bad", "purpose": "test"},
            failing_query, lambda sql: True, {},
        )
        assert result["status"] == "error"
        assert "ambiguous" in result["error"]

    def test_query_database_empty_query(self):
        result, _ = skills.dispatch_tool(
            "query_database",
            {"query": "", "purpose": "test"},
            lambda **k: pd.DataFrame(), lambda sql: True, {},
        )
        assert result["status"] == "error"

    def test_create_excel_returns_artifact(self):
        result, artifact = skills.dispatch_tool(
            "create_excel_artifact",
            {
                "filename": "test.xlsx",
                "title": "Test",
                "summary": "summary",
                "sheets": {"S": [{"a": 1}]},
            },
            None, None, {},
        )
        assert result["status"] == "ok"
        assert "artifact_id" in result
        assert artifact is not None
        assert artifact["type"] == "excel"

    def test_create_word_returns_artifact(self):
        result, artifact = skills.dispatch_tool(
            "create_word_artifact",
            {
                "filename": "r.docx",
                "title": "R",
                "summary": "s",
                "sections": [{"heading": "H", "body": "b"}],
            },
            None, None, {},
        )
        assert result["status"] == "ok"
        assert artifact["type"] == "word"

    def test_unknown_tool(self):
        result, _ = skills.dispatch_tool("unknown_tool", {}, None, None, {})
        assert result["status"] == "error"


# ===================================================================
# Format helpers
# ===================================================================

class TestFormatSize:
    def test_bytes(self):
        assert skills.format_size(100) == "100 B"

    def test_kb(self):
        assert skills.format_size(2048) == "2.0 KB"

    def test_mb(self):
        assert skills.format_size(5 * 1024 * 1024) == "5.00 MB"
