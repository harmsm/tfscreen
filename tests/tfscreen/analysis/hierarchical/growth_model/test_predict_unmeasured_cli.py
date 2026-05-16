import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli import (
    _read_lines,
    predict_unmeasured_cli,
    main,
)


# ---------------------------------------------------------------------------
# _read_lines
# ---------------------------------------------------------------------------

def test_read_lines_basic(tmp_path):
    f = tmp_path / "values.txt"
    f.write_text("wt\nM42I\nK84L\n")
    assert _read_lines(str(f)) == ["wt", "M42I", "K84L"]


def test_read_lines_skips_blank_and_comments(tmp_path):
    f = tmp_path / "values.txt"
    f.write_text("# comment\n\nwt\n  \nM42I\n# another comment\nK84L\n")
    assert _read_lines(str(f)) == ["wt", "M42I", "K84L"]


def test_read_lines_all_ignored(tmp_path):
    f = tmp_path / "values.txt"
    f.write_text("# only comments\n\n   \n")
    assert _read_lines(str(f)) == []


# ---------------------------------------------------------------------------
# predict_unmeasured_cli (the core function, independent of argparse)
# ---------------------------------------------------------------------------

def _make_dummy_result():
    return pd.DataFrame({
        "genotype": ["wt", "M42I"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [0.0, 0.0],
        "theta_median": [0.5, 0.4],
    })


def test_predict_unmeasured_cli_writes_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dummy_df = _make_dummy_result()

    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.read_configuration") as mock_cfg, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.extract_theta_unmeasured") as mock_extract:
        mock_cfg.return_value = (MagicMock(), {})
        mock_extract.return_value = dummy_df

        predict_unmeasured_cli(
            config_file="config.yaml",
            posterior_file="post.h5",
            titrant_name=["IPTG"],
            titrant_conc=[0.0, 10.0, 100.0],
            genotypes=["wt", "M42I"],
            out_prefix="mytest",
        )

    out = tmp_path / "mytest_theta_unmeasured.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == list(dummy_df.columns)


def test_predict_unmeasured_cli_broadcasts_single_titrant_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.read_configuration") as mock_cfg, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.extract_theta_unmeasured") as mock_extract:
        mock_cfg.return_value = (MagicMock(), {})
        mock_extract.return_value = _make_dummy_result()

        predict_unmeasured_cli(
            config_file="config.yaml",
            posterior_file="post.h5",
            titrant_name=["IPTG"],
            titrant_conc=[0.0, 10.0, 100.0],
            genotypes=["wt"],
            out_prefix="out",
        )

    _, kwargs = mock_extract.call_args
    titrant_df = kwargs["manual_titrant_df"]
    assert list(titrant_df["titrant_name"]) == ["IPTG", "IPTG", "IPTG"]
    assert list(titrant_df["titrant_conc"]) == [0.0, 10.0, 100.0]


def test_predict_unmeasured_cli_mismatched_titrant_lengths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.read_configuration") as mock_cfg:
        mock_cfg.return_value = (MagicMock(), {})

        with pytest.raises(ValueError, match="titrant_name has"):
            predict_unmeasured_cli(
                config_file="config.yaml",
                posterior_file="post.h5",
                titrant_name=["IPTG", "ATC"],
                titrant_conc=[0.0, 10.0, 100.0],
                genotypes=["wt"],
                out_prefix="out",
            )


# ---------------------------------------------------------------------------
# main() — argparse layer
# ---------------------------------------------------------------------------

def _run_main(argv, mock_cli):
    """Helper: patch predict_unmeasured_cli and run main() with argv."""
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_unmeasured_cli.predict_unmeasured_cli",
               mock_cli), \
         patch("sys.argv", ["tfs-predict-unmeasured"] + argv):
        main()


def test_main_inline_titrant_conc(tmp_path):
    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc", "0", "10", "100",
        "--genotypes", "wt", "M42I",
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["titrant_conc"] == [0.0, 10.0, 100.0]
    assert kwargs["titrant_name"] == ["IPTG"]
    assert kwargs["genotypes"] == ["wt", "M42I"]


def test_main_titrant_conc_from_file(tmp_path):
    conc_file = tmp_path / "concs.txt"
    conc_file.write_text("0\n10\n100\n")

    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc_file", str(conc_file),
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["titrant_conc"] == [0.0, 10.0, 100.0]
    assert kwargs["genotypes"] is None


def test_main_genotypes_from_file(tmp_path):
    geno_file = tmp_path / "genos.txt"
    geno_file.write_text("wt\nM42I\nK84L\n")

    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc", "0", "10",
        "--genotypes_file", str(geno_file),
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["genotypes"] == ["wt", "M42I", "K84L"]


def test_main_both_file_sources(tmp_path):
    conc_file = tmp_path / "concs.txt"
    conc_file.write_text("0.0\n50.0\n")
    geno_file = tmp_path / "genos.txt"
    geno_file.write_text("# header\nwt\nM42I\n")

    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc_file", str(conc_file),
        "--genotypes_file", str(geno_file),
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["titrant_conc"] == [0.0, 50.0]
    assert kwargs["genotypes"] == ["wt", "M42I"]
    assert kwargs["config_file"] == "config.yaml"
    assert kwargs["posterior_file"] == "post.h5"


def test_main_titrant_conc_and_file_are_mutually_exclusive(tmp_path):
    conc_file = tmp_path / "concs.txt"
    conc_file.write_text("0\n10\n")

    with pytest.raises(SystemExit):
        with patch("sys.argv", [
            "tfs-predict-unmeasured",
            "config.yaml", "post.h5",
            "--titrant_name", "IPTG",
            "--titrant_conc", "0", "10",
            "--titrant_conc_file", str(conc_file),
        ]):
            main()


def test_main_genotypes_and_file_are_mutually_exclusive(tmp_path):
    geno_file = tmp_path / "genos.txt"
    geno_file.write_text("wt\n")

    with pytest.raises(SystemExit):
        with patch("sys.argv", [
            "tfs-predict-unmeasured",
            "config.yaml", "post.h5",
            "--titrant_name", "IPTG",
            "--titrant_conc", "0",
            "--genotypes", "wt",
            "--genotypes_file", str(geno_file),
        ]):
            main()


def test_main_missing_titrant_conc_exits(tmp_path):
    with pytest.raises(SystemExit):
        with patch("sys.argv", [
            "tfs-predict-unmeasured",
            "config.yaml", "post.h5",
            "--titrant_name", "IPTG",
        ]):
            main()


def test_main_missing_titrant_name_exits(tmp_path):
    with pytest.raises(SystemExit):
        with patch("sys.argv", [
            "tfs-predict-unmeasured",
            "config.yaml", "post.h5",
            "--titrant_conc", "0", "10",
        ]):
            main()


def test_main_out_prefix_forwarded(tmp_path):
    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc", "0",
        "--out_prefix", "my_output",
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["out_prefix"] == "my_output"


def test_main_default_out_prefix(tmp_path):
    mock_cli = MagicMock()
    _run_main([
        "config.yaml", "post.h5",
        "--titrant_name", "IPTG",
        "--titrant_conc", "0",
    ], mock_cli)

    _, kwargs = mock_cli.call_args
    assert kwargs["out_prefix"] == "tfs_theta_pred"
