"""
Unit tests for tfscreen.util.grid_utils.

The relativize_node / relativize_config_paths / relativize_template_vars
functions are already covered in
tests/tfscreen/tfmodel/scripts/test_setup_grid_cli.py, so this file focuses
on the remaining helpers: sanitize, make_jinja_env, and make_run_name.
"""

import pytest

from tfscreen.util.grid_utils import sanitize, make_jinja_env, make_run_name


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_plain_word_unchanged(self):
        assert sanitize("hello") == "hello"

    def test_replaces_dot_with_underscore(self):
        assert sanitize("a.b") == "a_b"

    def test_replaces_space_with_underscore(self):
        assert sanitize("a b") == "a_b"

    def test_replaces_slash_with_underscore(self):
        assert sanitize("a/b") == "a_b"

    def test_collapses_consecutive_underscores(self):
        assert sanitize("a..b") == "a_b"
        assert sanitize("a  b") == "a_b"

    def test_strips_leading_trailing_underscores(self):
        assert sanitize("_hello_") == "hello"
        assert sanitize(".hello.") == "hello"

    def test_preserves_hyphens(self):
        assert sanitize("a-b") == "a-b"

    def test_preserves_digits(self):
        assert sanitize("abc123") == "abc123"

    def test_numeric_string(self):
        # Numbers should come through as-is
        assert sanitize(42) == "42"

    def test_empty_string_gives_empty(self):
        assert sanitize("") == ""

    def test_only_special_chars_gives_empty(self):
        assert sanitize("...") == ""


# ---------------------------------------------------------------------------
# make_jinja_env
# ---------------------------------------------------------------------------

class TestMakeJinjaEnv:
    def test_returns_jinja_environment(self):
        import jinja2
        env = make_jinja_env()
        assert isinstance(env, jinja2.Environment)

    def test_basename_filter_registered(self):
        env = make_jinja_env()
        assert "basename" in env.filters

    def test_basename_filter_works(self):
        env = make_jinja_env()
        tmpl = env.from_string("{{ path | basename }}")
        assert tmpl.render(path="/a/b/c.yaml") == "c.yaml"

    def test_strict_mode_raises_on_missing_var(self):
        import jinja2
        env = make_jinja_env(strict=True)
        tmpl = env.from_string("{{ missing }}")
        with pytest.raises(jinja2.UndefinedError):
            tmpl.render()

    def test_non_strict_mode_renders_empty_on_missing(self):
        env = make_jinja_env(strict=False)
        tmpl = env.from_string("{{ missing }}")
        # Should not raise; missing var renders as empty string
        result = tmpl.render()
        assert result == ""


# ---------------------------------------------------------------------------
# make_run_name
# ---------------------------------------------------------------------------

class TestMakeRunName:
    def test_prefix_always_present(self):
        name = make_run_name(None, {"x": "val"}, 0)
        assert name.startswith("run_0000")

    def test_index_zero_padded_to_4_digits(self):
        name = make_run_name(None, {"x": "a"}, 7)
        assert name.startswith("run_0007")

    def test_large_index(self):
        name = make_run_name(None, {"x": "a"}, 1234)
        assert name.startswith("run_1234")

    def test_no_template_uses_var_values(self):
        name = make_run_name(None, {"alpha": "foo", "beta": "bar"}, 1)
        assert "foo" in name
        assert "bar" in name

    def test_no_template_sanitizes_values(self):
        name = make_run_name(None, {"x": "a.b/c"}, 0)
        # Special chars should not appear raw
        assert "." not in name
        assert "/" not in name

    def test_with_template(self):
        name = make_run_name("{{ x }}__{{ y }}", {"x": "alpha", "y": "beta"}, 2)
        assert "alpha" in name
        assert "beta" in name

    def test_template_output_sanitized(self):
        name = make_run_name("{{ x }}", {"x": "val with spaces"}, 0)
        assert " " not in name

    def test_empty_suffix_returns_prefix_only(self):
        # If all var values are empty, suffix is empty → name == prefix
        name = make_run_name(None, {"x": ""}, 3)
        assert name == "run_0003"

    def test_bad_template_raises_value_error(self):
        with pytest.raises(ValueError, match="run_name template error"):
            make_run_name("{% for %}", {}, 0)
