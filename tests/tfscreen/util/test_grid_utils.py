"""
Unit tests for tfscreen.util.grid_utils: sanitize, make_jinja_env,
make_run_name and the input-staging helpers (InputStager,
check_no_outside_paths, stage_template_vars, render_run_template).
"""

import os

import pytest

from tfscreen.util.grid_utils import (
    INPUTS_DIRNAME,
    InputStager,
    check_no_outside_paths,
    make_jinja_env,
    make_run_name,
    render_run_template,
    sanitize,
    stage_template_vars,
)


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


# ---------------------------------------------------------------------------
# InputStager
# ---------------------------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


class TestInputStager:
    def test_copies_once_and_returns_run_relative_path(self, tmp_path):
        src = _write(tmp_path / "data" / "growth.csv", "g")
        stager = InputStager(tmp_path / "grid", "tfs-test")
        rel = stager.stage(src, "x")
        assert rel == os.path.join("..", INPUTS_DIRNAME, "growth.csv")
        assert stager.stage(src, "x") == rel
        assert os.listdir(tmp_path / "grid" / INPUTS_DIRNAME) == ["growth.csv"]
        run_dir = tmp_path / "grid" / "run_0001"
        run_dir.mkdir()
        assert open(os.path.join(run_dir, rel)).read() == "g"

    def test_same_name_different_file_gets_suffix(self, tmp_path):
        a = _write(tmp_path / "a" / "growth.csv", "a")
        b = _write(tmp_path / "b" / "growth.csv", "b")
        stager = InputStager(tmp_path / "grid", "tfs-test")
        assert stager.stage(a, "x").endswith("growth.csv")
        rel_b = stager.stage(b, "x")
        assert rel_b.endswith("growth_2.csv")
        assert open(tmp_path / "grid" / INPUTS_DIRNAME / "growth_2.csv").read() == "b"

    def test_never_overwrites_changed_existing_copy(self, tmp_path):
        old = _write(tmp_path / "grid" / INPUTS_DIRNAME / "growth.csv", "old")
        src = _write(tmp_path / "data" / "growth.csv", "new")
        rel = InputStager(tmp_path / "grid", "tfs-test").stage(src, "x")
        assert rel.endswith("growth_2.csv")
        assert open(old).read() == "old"

    def test_reuses_identical_existing_copy(self, tmp_path):
        _write(tmp_path / "grid" / INPUTS_DIRNAME / "growth.csv", "same")
        src = _write(tmp_path / "data" / "growth.csv", "same")
        rel = InputStager(tmp_path / "grid", "tfs-test").stage(src, "x")
        assert rel.endswith(os.sep + "growth.csv")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="my key"):
            InputStager(tmp_path, "tfs-test").stage(str(tmp_path / "no.csv"), "my key")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="directory"):
            InputStager(tmp_path, "tfs-test").stage(str(tmp_path), "my key")

    def test_dry_run_copies_nothing(self, tmp_path):
        src = _write(tmp_path / "data" / "growth.csv", "g")
        InputStager(tmp_path / "grid", "tfs-test", dry_run=True).stage(src, "x")
        assert not os.path.exists(tmp_path / "grid")


# ---------------------------------------------------------------------------
# check_no_outside_paths
# ---------------------------------------------------------------------------

class TestCheckNoOutsidePaths:
    def test_known_key_allowed(self, tmp_path):
        f = _write(tmp_path / "g.csv", "x")
        check_no_outside_paths({"growth_df": f}, [("growth_df",)], str(tmp_path), "")

    def test_unknown_absolute_file_raises(self, tmp_path):
        f = _write(tmp_path / "g.csv", "x")
        with pytest.raises(ValueError, match="a.b.*HINT"):
            check_no_outside_paths({"a": {"b": f}}, [], str(tmp_path), "HINT")

    def test_unknown_relative_file_raises(self, tmp_path):
        _write(tmp_path / "g.csv", "x")
        with pytest.raises(ValueError, match="g.csv"):
            check_no_outside_paths({"a": ["g.csv"]}, [], str(tmp_path), "")

    def test_per_key_source_dirs(self, tmp_path):
        _write(tmp_path / "base" / "g.csv", "x")
        cfg = {"a": "g.csv", "b": "g.csv"}
        dirs = {"a": str(tmp_path), "b": str(tmp_path / "base")}
        with pytest.raises(ValueError, match="'b'"):
            check_no_outside_paths(cfg, [], dirs, "")

    def test_non_paths_pass(self, tmp_path):
        check_no_outside_paths({"a": "linear", "b": 1, "c": [0.1, None]},
                               [], str(tmp_path), "")


# ---------------------------------------------------------------------------
# stage_template_vars / render_run_template
# ---------------------------------------------------------------------------

class TestStageTemplateVars:
    def test_file_values_staged_others_untouched(self, tmp_path):
        _write(tmp_path / "genos.txt", "wt")
        stager = InputStager(tmp_path / "grid", "tfs-test")
        out = stage_template_vars(
            {"f": "genos.txt", "seed": 1, "label": "x", "here": "."},
            str(tmp_path), stager)
        assert out == {"f": os.path.join("..", INPUTS_DIRNAME, "genos.txt"),
                       "seed": 1, "label": "x", "here": "."}

    def test_directory_value_raises(self, tmp_path):
        (tmp_path / "d").mkdir()
        with pytest.raises(ValueError, match="directory"):
            stage_template_vars({"f": "d"}, str(tmp_path),
                                InputStager(tmp_path, "tfs-test"))

    def test_render_names_run_on_undefined(self):
        tmpl = make_jinja_env(strict=True).from_string("{{ missing }}")
        with pytest.raises(ValueError, match="run_0007"):
            render_run_template(tmpl, {}, "run_0007")
