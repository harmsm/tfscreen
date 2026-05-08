import os
import pytest
import yaml
import tempfile
from tfscreen.util.io.read_yaml import read_yaml, _normalize_types

class TestNormalizeTypes:
    def test_scientific_notation_strings(self):
        # Strings that look like sci notation should be converted
        assert _normalize_types("1.0e2") == 100
        assert isinstance(_normalize_types("1.0e2"), int) # 100.0 -> 100 (int)
        
        assert _normalize_types("1.5e1") == 15
        assert isinstance(_normalize_types("1.5e1"), int)
        
        assert _normalize_types("1.5e-1") == 0.15
        assert isinstance(_normalize_types("1.5e-1"), float)

    def test_float_to_int(self):
        # Floats that are whole numbers should become ints
        assert _normalize_types(10.0) == 10
        assert isinstance(_normalize_types(10.0), int)
        
        assert _normalize_types(10.5) == 10.5
        assert isinstance(_normalize_types(10.5), float)

    def test_recursion(self):
        # Should work on lists and dicts
        data = {
            "a": "1.0e2",
            "b": [10.0, "5.5", "2.0e1"],
            "c": {"d": 5.0}
        }
        expected = {
            "a": 100,
            "b": [10, "5.5", 20],
            "c": {"d": 5}
        }
        assert _normalize_types(data) == expected

    def test_other_types_remain(self):
        # Bools, regular strings, etc should be untouched
        data = ["string", True, None]
        assert _normalize_types(data) == ["string", True, None]

class TestReadYaml:
    def test_read_dict_returns_as_is(self):
        # If a dict is passed, it returns it directly (no normalization)
        data = {"a": 10.0}
        assert read_yaml(data) is data
        
    def test_read_file_success(self):
        content = """
        key: 1.0e2
        nested:
          val: 5.0
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            result = read_yaml(tmp_path)
            # Should have normalized types
            assert result['key'] == 100
            assert result['nested']['val'] == 5
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_file_not_found(self, capsys):
        result = read_yaml("nonexistent_file.yaml")
        assert result is None
        captured = capsys.readouterr()
        assert "Error: Configuration file not found" in captured.out

    def test_invalid_yaml(self, capsys):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write("key: : value") # Invalid YAML
            tmp_path = tmp.name
            
        try:
            result = read_yaml(tmp_path)
            assert result is None
            captured = capsys.readouterr()
            assert "Error parsing YAML file" in captured.out
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_override_keys(self):
        content = "key1: value1\nkey2: value2"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            result = read_yaml(tmp_path, override_keys={'key1': 'new_value'})
            assert result['key1'] == 'new_value'
            assert result['key2'] == 'value2'
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_override_keys_missing_raises(self):
        content = "key1: value1"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="override_keys has a key"):
                read_yaml(tmp_path, override_keys={'key2': 'new_value'})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestGrowthConditionBlocksValidation:
    """Tests for the growth / condition_blocks consistency check in read_yaml."""

    def _write_yaml(self, data):
        """Write a dict to a temp YAML file; caller must delete it."""
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
        # Remove "pre" so condition_pre has no entry
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
        cfg["growth"]["ghost"] = {"m": 0.0, "b": 0.0}   # extra
        del cfg["growth"]["pre"]                           # missing
        path = self._write_yaml(cfg)
        try:
            with pytest.raises(ValueError) as exc_info:
                read_yaml(path)
            msg = str(exc_info.value)
            assert "pre" in msg       # missing condition named
            assert "ghost" in msg     # extra key named
        finally:
            os.remove(path)

    def test_multiple_blocks_all_conditions_checked(self):
        # Two blocks; "sel2" is present in block but absent from growth
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
        # A condition used for both pre and sel requires only one entry in growth
        cfg = {
            "growth": {"neutral": {"m": 0.0, "b": 0.01}},
            "condition_blocks": [
                {"condition_pre": "neutral", "condition_sel": "neutral"},
            ],
        }
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)   # must not raise
        finally:
            os.remove(path)

    def test_no_growth_key_skips_validation(self):
        # Config without a growth block is not validated at all
        cfg = {
            "condition_blocks": [
                {"condition_pre": "pre", "condition_sel": "sel"},
            ]
        }
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)   # must not raise
        finally:
            os.remove(path)

    def test_no_condition_blocks_key_skips_validation(self):
        # Config with growth but no condition_blocks is not validated
        cfg = {"growth": {"sel": {"m": 0.0, "b": 0.0}}}
        path = self._write_yaml(cfg)
        try:
            read_yaml(path)   # must not raise
        finally:
            os.remove(path)
