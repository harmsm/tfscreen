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
