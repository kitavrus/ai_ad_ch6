"""Покрытие ранее непокрытых веток chatbot/cli.py."""

from llm_agent.chatbot.cli import parse_inline_command


class TestStrategyAliases:
    def test_sw_alias(self):
        result = parse_inline_command("/strategy sw")
        assert result.get("strategy") == "sliding_window"

    def test_sf_alias(self):
        result = parse_inline_command("/strategy sf")
        assert result.get("strategy") == "sticky_facts"

    def test_br_alias(self):
        result = parse_inline_command("/strategy br")
        assert result.get("strategy") == "branching"

    def test_valid_full_name(self):
        result = parse_inline_command("/strategy sliding_window")
        assert result.get("strategy") == "sliding_window"

    def test_unknown_strategy_no_update(self):
        result = parse_inline_command("/strategy unknown_xyz")
        assert "strategy" not in result


class TestShowfacts:
    def test_showfacts_sets_true(self):
        result = parse_inline_command("/showfacts")
        assert result.get("showfacts") is True


class TestSetfact:
    def test_colon_format(self):
        result = parse_inline_command("/setfact lang: Python")
        assert result.get("setfact") == {"key": "lang", "value": "Python"}

    def test_space_format(self):
        result = parse_inline_command("/setfact lang Python")
        assert result.get("setfact") == {"key": "lang", "value": "Python"}

    def test_empty_key_ignored(self):
        result = parse_inline_command("/setfact")
        assert "setfact" not in result

    def test_empty_value_ignored(self):
        result = parse_inline_command("/setfact lang:")
        assert "setfact" not in result

    def test_key_lowercased(self):
        result = parse_inline_command("/setfact Lang: Py")
        assert result["setfact"]["key"] == "lang"


class TestDelfact:
    def test_nonempty_value(self):
        result = parse_inline_command("/delfact lang")
        assert result.get("delfact") == "lang"

    def test_empty_value_ignored(self):
        result = parse_inline_command("/delfact")
        assert "delfact" not in result

    def test_value_lowercased(self):
        result = parse_inline_command("/delfact LANG")
        assert result.get("delfact") == "lang"


class TestBranchingCommands:
    def test_checkpoint(self):
        result = parse_inline_command("/checkpoint")
        assert result.get("checkpoint") is True

    def test_branch_with_name(self):
        result = parse_inline_command("/branch my-branch")
        assert result.get("branch") == "my-branch"

    def test_branch_without_name_auto_generates(self):
        result = parse_inline_command("/branch")
        assert result.get("branch", "").startswith("branch-")
        assert len(result["branch"]) > len("branch-")

    def test_switch_with_name(self):
        result = parse_inline_command("/switch variant-a")
        assert result.get("switch") == "variant-a"

    def test_switch_empty_ignored(self):
        result = parse_inline_command("/switch")
        assert "switch" not in result

    def test_branches(self):
        result = parse_inline_command("/branches")
        assert result.get("branches") is True


class TestMemoryCommands:
    def test_memshow_default(self):
        result = parse_inline_command("/memshow")
        assert result.get("memshow") == "all"

    def test_memshow_explicit(self):
        result = parse_inline_command("/memshow short_term")
        assert result.get("memshow") == "short_term"

    def test_memstats(self):
        result = parse_inline_command("/memstats")
        assert result.get("memstats") is True

    def test_memclear_default(self):
        result = parse_inline_command("/memclear")
        assert result.get("memclear") == "short_term"

    def test_memclear_explicit(self):
        result = parse_inline_command("/memclear all")
        assert result.get("memclear") == "all"

    def test_memsave_default(self):
        result = parse_inline_command("/memsave")
        assert result.get("memsave") == "all"

    def test_memload_default(self):
        result = parse_inline_command("/memload")
        assert result.get("memload") == "all"

    def test_settask_nonempty(self):
        result = parse_inline_command("/settask Build API")
        assert result.get("settask") == "Build API"

    def test_settask_empty_ignored(self):
        result = parse_inline_command("/settask")
        assert "settask" not in result

    def test_setpref_nonempty(self):
        result = parse_inline_command("/setpref=lang=Python")
        assert result.get("setpref") == "lang=Python"

    def test_setpref_empty_ignored(self):
        result = parse_inline_command("/setpref")
        assert "setpref" not in result

    def test_remember_nonempty(self):
        result = parse_inline_command("/remember Use black formatter")
        assert result.get("remember") == "Use black formatter"

    def test_remember_empty_ignored(self):
        result = parse_inline_command("/remember")
        assert "remember" not in result


class TestProfileCommand:
    def test_profile_show(self):
        result = parse_inline_command("/profile show")
        assert result.get("profile") == {"action": "show", "arg": ""}

    def test_profile_list(self):
        result = parse_inline_command("/profile list")
        assert result.get("profile") == {"action": "list", "arg": ""}

    def test_profile_name(self):
        result = parse_inline_command("/profile name Igor")
        assert result.get("profile") == {"action": "name", "arg": "Igor"}

    def test_profile_style(self):
        result = parse_inline_command("/profile style tone=formal")
        assert result.get("profile") == {"action": "style", "arg": "tone=formal"}

    def test_profile_constraint_add(self):
        result = parse_inline_command("/profile constraint add no emojis")
        assert result.get("profile") == {"action": "constraint", "arg": "add no emojis"}

    def test_profile_default_action_show(self):
        result = parse_inline_command("/profile")
        assert result.get("profile") == {"action": "show", "arg": ""}
