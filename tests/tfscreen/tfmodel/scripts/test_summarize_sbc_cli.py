"""Tests for summarize_sbc_cli.py — thin CLI wrapper around summarize_sbc."""
from unittest.mock import patch


class TestSummarizeSbcMain:

    def test_main_calls_generalized_main_with_summarize_sbc(self):
        with patch("tfscreen.tfmodel.scripts.summarize_sbc_cli.generalized_main") as mock_gm:
            from tfscreen.tfmodel.scripts import summarize_sbc_cli
            summarize_sbc_cli.main()

        mock_gm.assert_called_once()
        args, kwargs = mock_gm.call_args
        assert args[0] is summarize_sbc_cli.summarize_sbc

    def test_summarize_sbc_imported_from_analysis(self):
        from tfscreen.tfmodel.scripts.summarize_sbc_cli import summarize_sbc
        from tfscreen.tfmodel.analysis.error_calibration import summarize_sbc as canonical
        assert summarize_sbc is canonical

    def test_main_passes_manual_arg_types(self):
        """manual_arg_types must declare sbc_dir and out_prefix as str."""
        with patch("tfscreen.tfmodel.scripts.summarize_sbc_cli.generalized_main") as mock_gm:
            from tfscreen.tfmodel.scripts import summarize_sbc_cli
            summarize_sbc_cli.main()

        _, kwargs = mock_gm.call_args
        mat = kwargs.get("manual_arg_types", {})
        assert mat.get("sbc_dir") is str
        assert mat.get("out_prefix") is str

    def test_main_is_callable(self):
        from tfscreen.tfmodel.scripts.summarize_sbc_cli import main
        assert callable(main)
