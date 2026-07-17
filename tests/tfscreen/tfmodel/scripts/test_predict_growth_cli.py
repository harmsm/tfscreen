"""
Tests for predict_growth_cli.py — genotype/conc union semantics and --only_files.
"""
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.scripts.predict_growth_cli import predict_growth
from tfscreen.tfmodel.inference.checkpoint_io import resolve_param_file


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_growth_df(genotypes, titrant_concs):
    rows = []
    for g in genotypes:
        for c in titrant_concs:
            rows.append({"genotype": g, "titrant_name": "IPTG",
                         "titrant_conc": c, "ln_cfu": 10.0})
    return pd.DataFrame(rows)


def _write_lines(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(str(x) for x in lines) + "\n")


def _fake_load_posteriors(param_file, q_to_get=None):
    """Fake for load_posteriors: (q_dict, posteriors) with a known sample count."""
    return {}, {"param1": np.zeros((10, 1))}


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.growth_df = _make_growth_df(["wt", "A1B"], [0.0, 1.0])
    # tensor_shape's last dim is genotype count (2); other axes product = 1.
    # Needed by estimate_genotype_batch_size, which auto-sizing now calls
    # whenever genotype_batch_size is left at its default of None.
    orchestrator.growth_tm.tensor_shape = (1, 1, 1, 1, 1, 1, 2)
    return orchestrator


@pytest.fixture
def mock_predict(mock_orchestrator):
    """Patch read_configuration, resolve_param_file, and predict; return captured call kwargs."""
    calls = {}

    def fake_predict(**kwargs):
        calls.update(kwargs)
        genotypes = kwargs.get("genotypes") or mock_orchestrator.growth_df["genotype"].unique().tolist()
        concs = kwargs.get("titrant_conc") or mock_orchestrator.growth_df["titrant_conc"].unique().tolist()
        rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                 "q0.5": 10.0}
                for g in genotypes for c in concs]
        return pd.DataFrame(rows)

    with patch(
        "tfscreen.tfmodel.scripts"
        ".predict_growth_cli.read_configuration",
        return_value=(mock_orchestrator, {}),
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_growth_cli.resolve_param_file",
        side_effect=lambda pf, orchestrator, op: pf,  # pass-through
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_growth_cli.predict",
        side_effect=fake_predict,
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_growth_cli.load_posteriors",
        side_effect=_fake_load_posteriors,
    ):
        yield calls


# ---------------------------------------------------------------------------
# Default behaviour (no files)
# ---------------------------------------------------------------------------

class TestPredictGrowthDefaults:

    def test_no_files_passes_none_genotypes(self, mock_predict, tmp_path):
        """genotypes is resolved to the full training list (not None), since
        auto-sizing (the default when genotype_batch_size is unset) needs an
        explicit list to size batches against. Functionally equivalent to the
        old None sentinel, which predict() also interprets as "all genotypes"."""
        predict_growth("cfg.yaml", "post.h5", out_prefix=str(tmp_path / "out"))
        assert sorted(mock_predict["genotypes"]) == ["A1B", "wt"]

    def test_no_files_passes_none_concs(self, mock_predict, tmp_path):
        predict_growth("cfg.yaml", "post.h5", out_prefix=str(tmp_path / "out"))
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# Union semantics (only_files=False, the default)
# ---------------------------------------------------------------------------

class TestPredictGrowthUnion:

    def test_genotypes_file_unions_with_training(self, mock_predict, mock_orchestrator, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])  # novel genotype
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        assert "wt" in mock_predict["genotypes"]
        assert "A1B" in mock_predict["genotypes"]
        assert "C2D" in mock_predict["genotypes"]

    def test_genotypes_file_union_preserves_order(self, mock_predict, mock_orchestrator, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        # training genotypes come before file genotypes
        idx_c2d = mock_predict["genotypes"].index("C2D")
        assert idx_c2d > 0

    def test_concs_file_unions_with_training(self, mock_predict, mock_orchestrator, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [5.0])  # novel concentration
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       out_prefix=str(tmp_path / "out"))
        assert 0.0 in mock_predict["titrant_conc"]
        assert 1.0 in mock_predict["titrant_conc"]
        assert 5.0 in mock_predict["titrant_conc"]

    def test_duplicate_concs_not_repeated(self, mock_predict, mock_orchestrator, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [1.0])  # already in training
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       out_prefix=str(tmp_path / "out"))
        concs = mock_predict["titrant_conc"]
        assert concs.count(1.0) == 1

    def test_duplicate_genotypes_not_repeated(self, mock_predict, mock_orchestrator, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["wt"])  # already in training
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"].count("wt") == 1


# ---------------------------------------------------------------------------
# --only_files semantics
# ---------------------------------------------------------------------------

class TestPredictGrowthOnlyFiles:

    def test_only_files_genotypes_restricts_to_file(self, mock_predict, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["A1B"])
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"] == ["A1B"]
        assert "wt" not in mock_predict["genotypes"]

    def test_only_files_concs_restricts_to_file(self, mock_predict, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [5.0])
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["titrant_conc"] == [5.0]
        assert 0.0 not in mock_predict["titrant_conc"]
        assert 1.0 not in mock_predict["titrant_conc"]

    def test_only_files_no_file_falls_through_to_none(self, mock_predict, tmp_path):
        """With no genotypes_file, genotypes falls through to all training
        genotypes (resolved to an explicit list for auto-sizing), same as the
        old None-sentinel behavior predict() would have used anyway."""
        predict_growth("cfg.yaml", "post.h5",
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert sorted(mock_predict["genotypes"]) == ["A1B", "wt"]
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# titrant_names_file is restrict-only (not unioned)
# ---------------------------------------------------------------------------

class TestPredictGrowthTitrantNamesFilter:

    def test_titrant_names_file_filters_output_rows(self, mock_predict, mock_orchestrator, tmp_path):
        nf = str(tmp_path / "names.txt")
        _write_lines(nf, ["IPTG"])
        out = str(tmp_path / "out")
        predict_growth("cfg.yaml", "post.h5",
                       titrant_names_file=nf,
                       out_prefix=out)
        df = pd.read_csv(f"{out}.csv")
        assert set(df["titrant_name"].unique()) <= {"IPTG"}

    def test_titrant_names_file_does_not_affect_predict_call(self, mock_predict, tmp_path):
        nf = str(tmp_path / "names.txt")
        _write_lines(nf, ["IPTG"])
        predict_growth("cfg.yaml", "post.h5",
                       titrant_names_file=nf,
                       out_prefix=str(tmp_path / "out"))
        # titrant_names_file must not influence genotypes or concs passed to predict
        assert sorted(mock_predict["genotypes"]) == ["A1B", "wt"]
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# checkpoint (.pkl) param_file support
# ---------------------------------------------------------------------------

class TestPredictGrowthCheckpointInput:

    def _make_fixtures(self, mock_orchestrator, resolved_path="resolved.h5"):
        """Return patch stack that intercepts resolve_param_file."""
        def fake_predict(**kwargs):
            genotypes = kwargs.get("genotypes") or mock_orchestrator.growth_df["genotype"].unique().tolist()
            concs = kwargs.get("titrant_conc") or mock_orchestrator.growth_df["titrant_conc"].unique().tolist()
            rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                     "q0.5": 10.0}
                    for g in genotypes for c in concs]
            return pd.DataFrame(rows)

        return [
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_growth_cli.read_configuration",
                return_value=(mock_orchestrator, {}),
            ),
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_growth_cli.predict",
                side_effect=fake_predict,
            ),
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_growth_cli.load_posteriors",
                side_effect=_fake_load_posteriors,
            ),
        ]

    def test_pkl_param_file_calls_resolve(self, mock_orchestrator, tmp_path):
        """resolve_param_file is called when param_file ends with .pkl."""
        resolve_calls = []

        def fake_resolve(pf, orchestrator, op):
            resolve_calls.append(pf)
            return "resolved.h5"

        patches = self._make_fixtures(mock_orchestrator)
        with patches[0], patches[1], patches[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=fake_resolve,
        ):
            predict_growth("cfg.yaml", "myrun_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert resolve_calls == ["myrun_checkpoint.pkl"]

    def test_h5_param_file_calls_resolve(self, mock_orchestrator, tmp_path):
        """resolve_param_file is called for .h5 files too (pass-through)."""
        resolve_calls = []

        def fake_resolve(pf, orchestrator, op):
            resolve_calls.append(pf)
            return pf

        patches = self._make_fixtures(mock_orchestrator)
        with patches[0], patches[1], patches[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=fake_resolve,
        ):
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"))

        assert resolve_calls == ["post.h5"]

    def test_resolved_path_passed_to_predict(self, mock_orchestrator, tmp_path):
        """The path returned by resolve_param_file is what predict receives."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["param_posteriors"] = kwargs.get("param_posteriors")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                  "titrant_conc": [0.0], "q0.5": [10.0]})

        patches = self._make_fixtures(mock_orchestrator)
        with patches[0], patches[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.resolve_param_file",
            return_value="resolved_map.h5",
        ):
            predict_growth("cfg.yaml", "myrun_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["param_posteriors"] == "resolved_map.h5"

    def test_pkl_passes_point_est_q_to_get(self, mock_orchestrator, tmp_path):
        """q_to_get=[0.5] is passed to predict when param_file is .pkl."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "q0.5": [10.0]})

        patches = self._make_fixtures(mock_orchestrator)
        with patches[0], patches[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.resolve_param_file",
            return_value="resolved.h5",
        ):
            predict_growth("cfg.yaml", "run_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["q_to_get"] == [0.5]

    def test_h5_passes_none_q_to_get(self, mock_orchestrator, tmp_path):
        """q_to_get=None is passed to predict when param_file is .h5."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "q0.5": [10.0]})

        patches = self._make_fixtures(mock_orchestrator)
        with patches[0], patches[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=lambda pf, orchestrator, op: pf,
        ):
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["q_to_get"] is None


# ---------------------------------------------------------------------------
# Genotype batching
# ---------------------------------------------------------------------------

class TestPredictGrowthBatching:
    """Tests for --genotype_batch_size batching logic."""

    def _make_fake_predict(self, orchestrator):
        """Return a fake predict that records every call and returns a DataFrame."""
        all_calls = []

        def fake_predict(**kwargs):
            all_calls.append(dict(kwargs))
            genotypes = kwargs.get("genotypes") or orchestrator.growth_df["genotype"].unique().tolist()
            concs = kwargs.get("titrant_conc") or orchestrator.growth_df["titrant_conc"].unique().tolist()
            rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                     "q0.5": 10.0}
                    for g in genotypes for c in concs]
            return pd.DataFrame(rows)

        return fake_predict, all_calls

    def _patch_stack(self, mock_orchestrator, fake_predict):
        return [
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.read_configuration",
                  return_value=(mock_orchestrator, {})),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.resolve_param_file",
                  side_effect=lambda pf, orchestrator, op: pf),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.predict",
                  side_effect=fake_predict),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.load_posteriors",
                  side_effect=_fake_load_posteriors),
        ]

    def test_no_batching_when_batch_size_none(self, mock_orchestrator, tmp_path):
        """genotype_batch_size=None → predict called exactly once."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=None)
        assert len(all_calls) == 1

    def test_batch_size_larger_than_total_single_call(self, mock_orchestrator, tmp_path):
        """genotype_batch_size > n_genotypes → still a single predict call."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1000)
        assert len(all_calls) == 1

    def test_batch_size_1_two_genotypes_two_calls(self, mock_orchestrator, tmp_path):
        """genotype_batch_size=1 with 2 training genotypes → 2 predict calls."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        assert len(all_calls) == 2

    def test_batch_each_call_gets_one_genotype(self, mock_orchestrator, tmp_path):
        """With batch_size=1, each predict call receives exactly 1 genotype."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        for call in all_calls:
            assert len(call["genotypes"]) == 1

    def test_batch_covers_all_genotypes(self, mock_orchestrator, tmp_path):
        """All training genotypes appear across the batched predict calls."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        seen = [call["genotypes"][0] for call in all_calls]
        expected = mock_orchestrator.growth_df["genotype"].unique().tolist()
        assert sorted(seen) == sorted(expected)

    def test_batch_results_concatenated_in_csv(self, mock_orchestrator, tmp_path):
        """Output CSV contains rows for every genotype across all batches."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=1)
        df = pd.read_csv(f"{out}.csv")
        assert set(df["genotype"].unique()) == {"wt", "A1B"}

    def test_batch_resolves_none_genotypes_from_orchestrator(self, mock_orchestrator, tmp_path):
        """When genotypes would be None, batching resolves from orchestrator.growth_df."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            # No genotypes_file → genotypes=None without batching; with batching
            # it must be resolved so the list can be split.
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        # Each call must have an explicit genotypes list, not None.
        for call in all_calls:
            assert call.get("genotypes") is not None

    def test_batch_with_file_genotypes_splits_file_list(self, mock_orchestrator, tmp_path):
        """When only_files=True and a genotypes file is given, batching splits the file list."""
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["wt", "A1B"])
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           genotypes_file=gf,
                           only_files=True,
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        assert len(all_calls) == 2
        seen = [call["genotypes"][0] for call in all_calls]
        assert sorted(seen) == ["A1B", "wt"]

    def test_batch_in_training_data_correct_across_batches(self, mock_orchestrator, tmp_path):
        """in_training_data column is correct for rows from all batches."""
        fake_predict, _ = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=1)
        df = pd.read_csv(f"{out}.csv")
        # Both genotypes in training data → all rows should be in_training_data=1
        assert (df["in_training_data"] == 1).all()


# ---------------------------------------------------------------------------
# Spiked-genotype augmentation during batching
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_orchestrator_binding():
    """Orchestrator with 'wt' as the only binding genotype (has theta_obs data)."""
    orchestrator = MagicMock()
    orchestrator.growth_df = _make_growth_df(["wt", "A1B"], [0.0, 1.0])
    orchestrator.binding_df = pd.DataFrame({
        "genotype": ["wt", "wt"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [0.0, 1.0],
        "theta_obs": [0.1, 0.9],
        "theta_std": [0.01, 0.01],
    })
    return orchestrator


class TestPredictGrowthBatchingSpiked:
    """Tests for binding-genotype augmentation when batching."""

    def _make_fake_predict(self, orchestrator):
        all_calls = []

        def fake_predict(**kwargs):
            all_calls.append(dict(kwargs))
            genotypes = kwargs.get("genotypes") or orchestrator.growth_df["genotype"].unique().tolist()
            concs = kwargs.get("titrant_conc") or orchestrator.growth_df["titrant_conc"].unique().tolist()
            rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                     "q0.5": 10.0}
                    for g in genotypes for c in concs]
            return pd.DataFrame(rows)

        return fake_predict, all_calls

    def _patch_stack(self, orchestrator, fake_predict):
        return [
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.read_configuration",
                  return_value=(orchestrator, {})),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.resolve_param_file",
                  side_effect=lambda pf, orch, op: pf),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.predict",
                  side_effect=fake_predict),
        ]

    def test_binding_geno_added_to_batch_missing_it(self, mock_orchestrator_binding, tmp_path):
        """The batch that doesn't contain 'wt' (binding genotype) gets it prepended."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_binding)
        patches = self._patch_stack(mock_orchestrator_binding, fake_predict)
        with patches[0], patches[1], patches[2]:
            # batch_size=1 → batch1=["wt"], batch2=["A1B"]; "wt" must appear in batch2 call
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        assert len(all_calls) == 2
        # batch2 is the "A1B" batch; "wt" (binding genotype) must be included
        a1b_call = next(c for c in all_calls if "A1B" in c["genotypes"])
        assert "wt" in a1b_call["genotypes"]

    def test_binding_geno_rows_stripped_from_non_native_batch(self, mock_orchestrator_binding, tmp_path):
        """Rows for extra binding genotypes are removed from the batch result."""
        fake_predict, _ = self._make_fake_predict(mock_orchestrator_binding)
        patches = self._patch_stack(mock_orchestrator_binding, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=1)
        df = pd.read_csv(f"{out}.csv")
        # "wt" should appear once per titrant_conc, not twice (no duplication)
        wt_rows = df[df["genotype"] == "wt"]
        expected_wt_rows = len(mock_orchestrator_binding.growth_df[
            mock_orchestrator_binding.growth_df["genotype"] == "wt"
        ]["titrant_conc"].unique())
        assert len(wt_rows) == expected_wt_rows

    def test_binding_geno_appears_exactly_once_in_output(self, mock_orchestrator_binding, tmp_path):
        """Binding genotype rows appear exactly once in the final CSV."""
        fake_predict, _ = self._make_fake_predict(mock_orchestrator_binding)
        patches = self._patch_stack(mock_orchestrator_binding, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=1)
        df = pd.read_csv(f"{out}.csv")
        wt_rows = df[df["genotype"] == "wt"]
        assert len(wt_rows) == wt_rows.drop_duplicates(
            subset=["genotype", "titrant_name", "titrant_conc"]
        ).shape[0]

    def test_all_genotypes_present_in_output_with_binding(self, mock_orchestrator_binding, tmp_path):
        """Both binding and library genotypes appear in the output CSV."""
        fake_predict, _ = self._make_fake_predict(mock_orchestrator_binding)
        patches = self._patch_stack(mock_orchestrator_binding, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=1)
        df = pd.read_csv(f"{out}.csv")
        assert set(df["genotype"].unique()) == {"wt", "A1B"}

    def test_no_augmentation_when_no_binding_df(self, mock_orchestrator, tmp_path):
        """When binding_df is a MagicMock (no real data), no augmentation happens."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator)
        patches = self._patch_stack(mock_orchestrator, fake_predict)
        with patches[0], patches[1], patches[2]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        # Each call should get exactly the original batch (1 genotype), no augmentation
        for call in all_calls:
            assert len(call["genotypes"]) == 1

    def test_binding_geno_not_in_requested_genotypes_still_augments(
            self, mock_orchestrator_binding, tmp_path):
        """Binding genotypes are added even when not in the user-requested list."""
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["A1B"])  # only the non-binding genotype
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_binding)
        patches = self._patch_stack(mock_orchestrator_binding, fake_predict)
        with patches[0], patches[1], patches[2]:
            predict_growth("cfg.yaml", "post.h5",
                           genotypes_file=gf,
                           only_files=True,
                           out_prefix=str(tmp_path / "out"),
                           genotype_batch_size=1)
        # Only 1 genotype in list → single call (no batching), but even if
        # batched, "wt" would be added and stripped.  The CSV should only
        # contain A1B rows.
        df = pd.read_csv(f"{tmp_path / 'out'}.csv")
        assert set(df["genotype"].unique()) == {"A1B"}


# ---------------------------------------------------------------------------
# --subset_genotypes: single memory-fit block for fast correlation checks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_orchestrator_subset():
    """Orchestrator with 6 training genotypes, 'wt' binding, 'A1B' spiked."""
    orchestrator = MagicMock()
    genos = ["wt", "A1B", "C2D", "E3F", "G4H", "I5K"]
    orchestrator.growth_df = _make_growth_df(genos, [0.0, 1.0])
    orchestrator.growth_tm.tensor_shape = (1, 1, 1, 1, 1, 1, len(genos))
    orchestrator.binding_df = pd.DataFrame({
        "genotype": ["wt", "wt"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [0.0, 1.0],
        "theta_obs": [0.1, 0.9],
        "theta_std": [0.01, 0.01],
    })
    # settings is a real dict so .get("spiked_genotypes") works.
    orchestrator.settings = {"spiked_genotypes": ["A1B"]}
    return orchestrator


class TestPredictGrowthSubset:
    """Tests for --subset_genotypes single-block sampling."""

    def _make_fake_predict(self, orchestrator):
        all_calls = []

        def fake_predict(**kwargs):
            all_calls.append(dict(kwargs))
            genotypes = kwargs.get("genotypes") or orchestrator.growth_df["genotype"].unique().tolist()
            concs = kwargs.get("titrant_conc") or orchestrator.growth_df["titrant_conc"].unique().tolist()
            rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                     "q0.5": 10.0}
                    for g in genotypes for c in concs]
            return pd.DataFrame(rows)

        return fake_predict, all_calls

    def _patch_stack(self, orchestrator, fake_predict):
        return [
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.read_configuration",
                  return_value=(orchestrator, {})),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.resolve_param_file",
                  side_effect=lambda pf, orch, op: pf),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.predict",
                  side_effect=fake_predict),
            patch("tfscreen.tfmodel.scripts.predict_growth_cli.load_posteriors",
                  side_effect=_fake_load_posteriors),
        ]

    def test_subset_single_predict_call(self, mock_orchestrator_subset, tmp_path):
        """Subset mode issues exactly one predict() call (no batching loop)."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=3)
        assert len(all_calls) == 1

    def test_subset_size_capped_at_block(self, mock_orchestrator_subset, tmp_path):
        """The predicted genotype count does not exceed the block size."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=3)
        assert len(all_calls[0]["genotypes"]) == 3

    def test_subset_always_includes_binding_and_spiked(self, mock_orchestrator_subset, tmp_path):
        """Binding ('wt') and spiked ('A1B') genotypes are always in the block."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=3,
                           subset_seed=0)
        genos = set(all_calls[0]["genotypes"])
        assert "wt" in genos      # binding
        assert "A1B" in genos     # spiked

    def test_subset_seed_is_deterministic(self, mock_orchestrator_subset, tmp_path):
        """Same subset_seed → identical sampled block across runs."""
        results = []
        for _ in range(2):
            fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
            patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
            with patches[0], patches[1], patches[2], patches[3]:
                predict_growth("cfg.yaml", "post.h5",
                               out_prefix=str(tmp_path / "out"),
                               subset_genotypes=True,
                               genotype_batch_size=3,
                               subset_seed=42)
            results.append(sorted(all_calls[0]["genotypes"]))
        assert results[0] == results[1]

    def test_subset_different_seeds_can_differ(self, mock_orchestrator_subset, tmp_path):
        """Different seeds draw different random members (the two mandatory
        genotypes are fixed; the one random slot should vary across seeds)."""
        sampled = set()
        for seed in range(6):
            fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
            patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
            with patches[0], patches[1], patches[2], patches[3]:
                predict_growth("cfg.yaml", "post.h5",
                               out_prefix=str(tmp_path / "out"),
                               subset_genotypes=True,
                               genotype_batch_size=3,
                               subset_seed=seed)
            # the non-mandatory member of the block
            extra = set(all_calls[0]["genotypes"]) - {"wt", "A1B"}
            sampled |= extra
        assert len(sampled) > 1

    def test_subset_block_larger_than_total_returns_all(self, mock_orchestrator_subset, tmp_path):
        """Block >= number of genotypes → every genotype is predicted."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=100)
        assert set(all_calls[0]["genotypes"]) == set(
            mock_orchestrator_subset.growth_df["genotype"].unique())

    def test_subset_auto_size_predicts_single_block(self, mock_orchestrator_subset, tmp_path):
        """With no explicit batch size, subset mode auto-sizes and still issues
        a single predict call (auto block on CPU is large → all genotypes)."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True)
        assert len(all_calls) == 1

    def test_subset_keeps_file_genotypes(self, mock_orchestrator_subset, tmp_path):
        """File-specified genotypes are always kept in the block. With a block
        of 3 and 3 mandatory genotypes (binding + spiked + file), no random
        members are drawn."""
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["G4H"])
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           genotypes_file=gf,
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=3)
        assert set(all_calls[0]["genotypes"]) == {"wt", "A1B", "G4H"}

    def test_subset_mandatory_exceeds_block_still_single_call(self, mock_orchestrator_subset, tmp_path):
        """When mandatory genotypes exceed the block, all are still predicted in
        one call (binding genotypes cannot be dropped)."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"),
                           subset_genotypes=True,
                           genotype_batch_size=1)
        assert len(all_calls) == 1
        genos = set(all_calls[0]["genotypes"])
        # both mandatory genotypes present despite block of 1
        assert {"wt", "A1B"} <= genos

    def test_subset_in_training_data_column(self, mock_orchestrator_subset, tmp_path):
        """Sampled genotypes are all from training data → in_training_data==1."""
        fake_predict, _ = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           subset_genotypes=True,
                           genotype_batch_size=3,
                           subset_seed=0)
        df = pd.read_csv(f"{out}.csv")
        assert (df["in_training_data"] == 1).all()

    def test_no_subset_predicts_all(self, mock_orchestrator_subset, tmp_path):
        """subset_genotypes=False (default) predicts every genotype."""
        fake_predict, all_calls = self._make_fake_predict(mock_orchestrator_subset)
        patches = self._patch_stack(mock_orchestrator_subset, fake_predict)
        out = str(tmp_path / "out")
        with patches[0], patches[1], patches[2], patches[3]:
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=out,
                           genotype_batch_size=100)
        df = pd.read_csv(f"{out}.csv")
        assert set(df["genotype"].unique()) == set(
            mock_orchestrator_subset.growth_df["genotype"].unique())
