import os
import pytest
import yaml
import tempfile
from tfscreen.util.io.read_yaml import read_yaml, _normalize_types


class TestNormalizeTypes:
    def test_scientific_notation_string_to_float(self):
        # Quoted sci-notation strings are converted to float (not int).
        result = _normalize_types("1.0e2")
        assert result == 100.0
        assert isinstance(result, float)

        result = _normalize_types("1.5e1")
        assert result == 15.0
        assert isinstance(result, float)

        result = _normalize_types("1.5e-1")
        assert abs(result - 0.15) < 1e-10
        assert isinstance(result, float)

    def test_float_passthrough(self):
        # Floats are not coerced to int regardless of value.
        assert _normalize_types(10.0) == 10.0
        assert isinstance(_normalize_types(10.0), float)
        assert _normalize_types(10.5) == 10.5

    def test_int_passthrough(self):
        # Integers stay integers.
        assert _normalize_types(100) == 100
        assert isinstance(_normalize_types(100), int)

    def test_recursion(self):
        # Should work on lists and dicts; floats are not coerced to int.
        data = {
            "a": "1.0e2",        # quoted sci-notation string → float
            "b": [10.0, "5.5", "2.0e1"],
            "c": {"d": 5.0}      # plain float stays float
        }
        result = _normalize_types(data)
        assert result["a"] == 100.0
        assert isinstance(result["a"], float)
        assert result["b"][0] == 10.0          # float stays float
        assert isinstance(result["b"][0], float)
        assert result["b"][1] == "5.5"         # non-sci string unchanged
        assert result["b"][2] == 20.0
        assert isinstance(result["b"][2], float)
        assert result["c"]["d"] == 5.0
        assert isinstance(result["c"]["d"], float)

    def test_other_types_remain(self):
        data = ["string", True, None]
        assert _normalize_types(data) == ["string", True, None]


class TestReadYaml:
    def test_read_dict_returns_as_is(self):
        data = {"a": 10.0}
        assert read_yaml(data) is data

    def test_read_file_success(self):
        content = "key: 1.0e2\nnested:\n  val: 5.0\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = read_yaml(tmp_path)
            # PyYAML parses 1.0e2 as float(100.0); _normalize_types leaves it.
            assert result['key'] == 100.0
            assert isinstance(result['key'], float)
            assert result['nested']['val'] == 5.0
        finally:
            os.remove(tmp_path)

    def test_int_literal_stays_int(self):
        # PyYAML parses unquoted integers (including underscore form) as int.
        content = "count: 100_000\nother: 42\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = read_yaml(tmp_path)
            assert result['count'] == 100_000
            assert isinstance(result['count'], int)
            assert isinstance(result['other'], int)
        finally:
            os.remove(tmp_path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_yaml("nonexistent_file.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write("key: : value")  # invalid YAML
            tmp_path = tmp.name
        try:
            with pytest.raises(yaml.YAMLError):
                read_yaml(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_override_keys(self):
        content = "key1: value1\nkey2: value2\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = read_yaml(tmp_path, override_keys={'key1': 'new_value'})
            assert result['key1'] == 'new_value'
            assert result['key2'] == 'value2'
        finally:
            os.remove(tmp_path)

    def test_override_keys_missing_raises(self):
        content = "key1: value1\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            with pytest.raises(ValueError, match="override_keys"):
                read_yaml(tmp_path, override_keys={'key2': 'new_value'})
        finally:
            os.remove(tmp_path)


class TestGrowthConditionBlocksValidation:
    """Tests for the growth / condition_blocks consistency check in read_yaml."""

    def _write_yaml(self, data):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def _valid_config(self):
        return {
            "growth": {
                "sel": {"m": -0.01, "b": 0.005},
                "pre": {"m":  0.001, "b": 0.020},
            },
            "condition_blocks": [
                {"condition_pre": "pre", "condition_sel": "sel"},
            ],
        }

    def test_valid_config_passes(self):
        path = self._write_yaml(self._valid_config())
        try:
            cfg = read_yaml(path)
            assert cfg["growth"]["sel"]["m"] == -0.01
        finally:
            os.remove(path)

    def test_condition_missing_from_growth_raises(self):
        cfg = self._valid_config()
        del cfg["growth"]["pre"]
        path = self._write_yaml(cfg)
        try:
            with pytest.raises(ValueError, match="pre"):
                read_yaml(path)
        finally:
            os.remove(path)

    def test_extra_key_in_growth_raises(self):
        cfg = self._valid_config()
        cfg["growth"]["unused_condition"] = {"m": 0.0, "b": 0.0}
        path = self._write_yaml(cfg)
        try:
            with pytest.raises(ValueError, match="unused_condition"):
                read_yaml(path)
        finally:
            os.remove(path)

    def test_both_errors_reported_together(self):
        cfg = self._valid_config()
        cfg["growth"]["ghost"] = {"m": 0.0, "b": 0.0}
        del cfg["growth"]["pre"]
        path = self._write_yaml(cfg)
        try:
            with pytest.raises(ValueError) as exc_info:
                read_yaml(path)
            msg = str(exc_info.value)
            assert "pre" in msg
            assert "ghost" in msg
        finally:
            os.remove(path)

    def test_multiple_blocks_all_conditions_checked(self):
        cfg = self._valid_config()
        cfg["condition_blocks"].append(
            {"condition_pre": "pre", "condition_sel": "sel2"}
        )
        path = self._write_yaml(cfg)
        try:
            with pytest.raises(ValueError, match="sel2"):
                read_yaml(path)
        finally:
            os.remove(path)

    def test_same_condition_pre_and_sel_allowed(self):
        cfg = {
            "growth": {"neutral": {"m": 0.0, "b": 0.01}},
            "condition_blocks": [
                {"condition_pre": "neutral", "condition_sel": "neutral"},
            ],
        }
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)
        finally:
            os.remove(path)

    def test_no_growth_key_skips_validation(self):
        cfg = {
            "condition_blocks": [
                {"condition_pre": "pre", "condition_sel": "sel"},
            ]
        }
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)
        finally:
            os.remove(path)

    def test_no_condition_blocks_key_skips_validation(self):
        cfg = {"growth": {"sel": {"m": 0.0, "b": 0.0}}}
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)
        finally:
            os.remove(path)
