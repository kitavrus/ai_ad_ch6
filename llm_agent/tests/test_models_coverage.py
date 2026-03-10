"""Покрытие ранее непокрытых веток chatbot/models.py."""

from llm_agent.chatbot.models import StickyFacts, UserProfile


class TestStickyFactsMethods:
    def test_get_existing_key(self):
        sf = StickyFacts(facts={"lang": "Python"})
        assert sf.get("lang") == "Python"

    def test_get_missing_key_returns_none(self):
        sf = StickyFacts()
        assert sf.get("missing") is None

    def test_set_adds_key(self):
        sf = StickyFacts()
        sf.set("lang", "Python")
        assert sf.facts["lang"] == "Python"

    def test_set_overwrites_existing(self):
        sf = StickyFacts(facts={"lang": "Java"})
        sf.set("lang", "Python")
        assert sf.facts["lang"] == "Python"

    def test_update_from_message(self):
        sf = StickyFacts()
        sf.update_from_message("topic", "databases", msg_index=3)
        assert sf.facts["topic"] == "databases"

    def test_to_list_empty(self):
        sf = StickyFacts()
        assert sf.to_list() == []

    def test_to_list_returns_key_value_dicts(self):
        sf = StickyFacts(facts={"lang": "Python", "style": "formal"})
        lst = sf.to_list()
        assert {"key": "lang", "value": "Python"} in lst
        assert {"key": "style", "value": "formal"} in lst


class TestUserProfileToSystemPrompt:
    def test_format_included(self):
        p = UserProfile(format={"output": "markdown", "code_blocks": "always"})
        prompt = p.to_system_prompt()
        assert "Format:" in prompt
        assert "output=markdown" in prompt

    def test_custom_included(self):
        p = UserProfile(custom={"persona": "senior dev"})
        prompt = p.to_system_prompt()
        assert "Extra:" in prompt
        assert "persona=senior dev" in prompt

    def test_all_fields_combined(self):
        p = UserProfile(
            style={"tone": "formal"},
            format={"output": "plain"},
            constraints=["no emojis"],
            custom={"focus": "backend"},
        )
        prompt = p.to_system_prompt()
        assert "Style:" in prompt
        assert "Format:" in prompt
        assert "Constraints:" in prompt
        assert "Extra:" in prompt

    def test_empty_profile_empty_prompt(self):
        p = UserProfile()
        assert p.to_system_prompt() == ""
