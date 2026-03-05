"""Тесты для разбора /task команды в chatbot/cli.py."""

import pytest

from chatbot.cli import parse_inline_command


class TestTaskCommandParsing:
    def test_task_new(self):
        result = parse_inline_command("/task new Build a REST API")
        assert result == {"task": {"action": "new", "arg": "Build a REST API"}}

    def test_task_show(self):
        result = parse_inline_command("/task show")
        assert result == {"task": {"action": "show", "arg": ""}}

    def test_task_list(self):
        result = parse_inline_command("/task list")
        assert result == {"task": {"action": "list", "arg": ""}}

    def test_task_start(self):
        result = parse_inline_command("/task start")
        assert result == {"task": {"action": "start", "arg": ""}}

    def test_task_step_done(self):
        result = parse_inline_command("/task step done")
        assert result == {"task": {"action": "step", "arg": "done"}}

    def test_task_step_skip(self):
        result = parse_inline_command("/task step skip")
        assert result == {"task": {"action": "step", "arg": "skip"}}

    def test_task_step_fail(self):
        result = parse_inline_command("/task step fail database error")
        assert result == {"task": {"action": "step", "arg": "fail database error"}}

    def test_task_step_note(self):
        result = parse_inline_command("/task step note this was tricky")
        assert result == {"task": {"action": "step", "arg": "note this was tricky"}}

    def test_task_pause(self):
        result = parse_inline_command("/task pause")
        assert result == {"task": {"action": "pause", "arg": ""}}

    def test_task_resume_no_id(self):
        result = parse_inline_command("/task resume")
        assert result == {"task": {"action": "resume", "arg": ""}}

    def test_task_resume_with_id(self):
        result = parse_inline_command("/task resume abc123")
        assert result == {"task": {"action": "resume", "arg": "abc123"}}

    def test_task_done(self):
        result = parse_inline_command("/task done")
        assert result == {"task": {"action": "done", "arg": ""}}

    def test_task_fail_no_reason(self):
        result = parse_inline_command("/task fail")
        assert result == {"task": {"action": "fail", "arg": ""}}

    def test_task_fail_with_reason(self):
        result = parse_inline_command("/task fail out of time")
        assert result == {"task": {"action": "fail", "arg": "out of time"}}

    def test_task_load(self):
        result = parse_inline_command("/task load abc123def456")
        assert result == {"task": {"action": "load", "arg": "abc123def456"}}

    def test_task_delete(self):
        result = parse_inline_command("/task delete myid")
        assert result == {"task": {"action": "delete", "arg": "myid"}}

    def test_task_bare_returns_show(self):
        result = parse_inline_command("/task")
        assert result == {"task": {"action": "show", "arg": ""}}

    def test_task_action_lowercased(self):
        result = parse_inline_command("/task NEW Build something")
        assert result["task"]["action"] == "new"

    def test_task_with_spaces_in_description(self):
        result = parse_inline_command("/task new Implement user authentication with OAuth2")
        assert result["task"]["arg"] == "Implement user authentication with OAuth2"

    def test_profile_not_affected(self):
        result = parse_inline_command("/profile show")
        assert "task" not in result
        assert "profile" in result

    def test_task_key_in_result(self):
        result = parse_inline_command("/task list")
        assert "task" in result
        assert isinstance(result["task"], dict)
        assert "action" in result["task"]
        assert "arg" in result["task"]

    def test_task_result_no_id(self):
        result = parse_inline_command("/task result")
        assert result == {"task": {"action": "result", "arg": ""}}

    def test_task_result_with_id(self):
        result = parse_inline_command("/task result abc123")
        assert result == {"task": {"action": "result", "arg": "abc123"}}
