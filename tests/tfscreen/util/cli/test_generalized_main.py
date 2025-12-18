import argparse
import sys
import pytest
from unittest.mock import MagicMock
from tfscreen.util.cli.generalized_main import generalized_main

class TestGeneralizedMain:
    def test_required_args(self):
        # Function with required arg (no default)
        mock_fcn = MagicMock(__name__="test_func", __doc__="Test Doc")
        # Define signature via a real function because signature is inspected
        def real_func(a, b): pass
        
        # We need to wrap generalized_main behavior or just pass real_func 
        # but we need to verify arguments passed.
        # Since generalized_main calls fcn(**args), we can replace fcn with a mock 
        # BUT inspection works on the passed object. inspected object must have signature.
        # MagicMock has signature if we configure it? Easier to just use real function side effect.
        
        start_result = {}
        def target_func(a, b):
            start_result['a'] = a
            start_result['b'] = b
            
        argv = ['1', 'val_b'] 
        # expected: a=1 (type str because no type inferred?), b='val_b'
        # Wait, if no default, type is None. argparse defaults to str.
        
        generalized_main(target_func, argv=argv)
        assert start_result['a'] == '1'
        assert start_result['b'] == 'val_b'

    def test_defaults_and_types(self):
        result = {}
        def target_func(x=10, y=2.5):
            result['x'] = x
            result['y'] = y
            
        # Call without args, should use defaults
        generalized_main(target_func, argv=[])
        assert result['x'] == 10
        assert result['y'] == 2.5
        
        # Call with overrides
        # x is int, y is float
        generalized_main(target_func, argv=['--x', '20', '--y', '0.5'])
        assert result['x'] == 20
        assert result['y'] == 0.5
        
    def test_bool_flags(self):
        result = {}
        def target_func(flag=False, inverse=True):
            result['flag'] = flag
            result['inverse'] = inverse
            
        # Defaults
        generalized_main(target_func, argv=[])
        assert result['flag'] is False
        assert result['inverse'] is True
        
        # Toggle
        generalized_main(target_func, argv=['--flag', '--inverse'])
        assert result['flag'] is True
        assert result['inverse'] is False

    def test_manual_overrides(self):
        result = {}
        def target_func(a, b=1):
            result['a'] = a
            result['b'] = b
            
        # Provide default for 'a', making it optional. Force typ for 'b' to str
        manual_defaults = {'a': 100}
        manual_types = {'b': str}
        
        generalized_main(target_func, 
                         argv=['--b', '20'], 
                         manual_arg_defaults=manual_defaults,
                         manual_arg_types=manual_types)
                         
        assert result['a'] == 100
        assert result['b'] == '20' # String now

    def test_manual_nargs(self):
        result = {}
        def target_func(items):
            result['items'] = items
            
        # items is required, but we want nargs='+'
        manual_nargs = {'items': '+'}
        
        generalized_main(target_func, 
                         argv=['i1', 'i2'], 
                         manual_arg_nargs=manual_nargs)
                         
        assert result['items'] == ['i1', 'i2']

    def test_missing_required_arg(self, capsys):
        def target_func(a): pass
        
        with pytest.raises(SystemExit):
            generalized_main(target_func, argv=[])
            
        # Argparse usually prints to stderr
        captured = capsys.readouterr()
        assert "error: the following arguments are required: a" in captured.err

    def test_bool_int_default_behavior(self):
        # Edge case: checking boolean handling logic
        # If type is bool, it checks default value to decide store_true vs store_false
        
        result = {}
        def target_func(v: bool = True):
            result['v'] = v
            
        # Should be store_false
        generalized_main(target_func, argv=['--v'])
        assert result['v'] is False
        
        generalized_main(target_func, argv=[])
        assert result['v'] is True

    def test_default_argv(self):
        # Test generalized_main when argv is None (uses sys.argv)
        import sys
        
        result = {}
        def target_func(a):
            result['a'] = a

        # Mock sys.argv
        with PatchSysArgv(['program.py', 'from_sys']):
            generalized_main(target_func)
            
        assert result['a'] == 'from_sys'

    def test_list_handling(self):
        """Test inferred type and nargs for list defaults."""
        result = {}
        def target_func(items=['a']):
            result['items'] = items
            
        # Should infer type=str and accept multiple args if manual_arg_nargs is set
        generalized_main(target_func, 
                         argv=['--items', 'x', 'y'], 
                         manual_arg_nargs={'items': '+'})
        assert result['items'] == ['x', 'y']

        # Test with empty list default (should default to str elements)
        result2 = {}
        def target_func2(items=[]):
            result2['items'] = items
        
        generalized_main(target_func2, 
                         argv=['--items', '1', '2'], 
                         manual_arg_nargs={'items': '+'})
        assert result2['items'] == ['1', '2'] # Stay as strings by default

from unittest.mock import patch

class PatchSysArgv:
    def __init__(self, argv):
        self.argv = argv
        self.patcher = patch.object(sys, 'argv', argv)

    def __enter__(self):
        self.patcher.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()
