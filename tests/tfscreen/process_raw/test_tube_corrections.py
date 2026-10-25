import hashlib
import json
import os
import textwrap

import pandas as pd
import pytest

from tfscreen.process_raw.scripts.correct_tubes_cli import correct_tubes
from tfscreen.process_raw.scripts.process_counts_cli import process_counts
from tfscreen.process_raw.tube_corrections import (
    DESIGN_COLUMNS,
    apply_corrections,
    correct_tube_table,
    read_corrections,
)

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "examples")
RAW = os.path.join(EXAMPLES, "process_raw")
TUBES = os.path.join(RAW, "tube_table.csv")
CORRECTIONS = os.path.join(RAW, "corrections.yaml")
CALIBRATION = os.path.join(EXAMPLES, "od600", "od600_calibration.yaml")

A = "pheS_pheSp4CP_0mM_t60"
B = "pheS_pheSp4CP_1mM_t60"
KAN = "kanR_kanRpkan_0mM_t120"


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _entry(**kw):
    e = {"id": "c1", "action": "swap", "tubes": [A, B], "reason": "QC",
         "by": "A. Researcher", "date": "2026-10-15"}
    e.update(kw)
    return e


def _write(tmp_path, entries, name="corr.yaml"):
    import yaml
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"corrections": entries}))
    return str(path)


def _apply(tmp_path, entries):
    corr = read_corrections(_write(tmp_path, entries))["corrections"]
    return correct_tube_table(pd.read_csv(TUBES), corr)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def test_swap_moves_design_columns_only(tmp_path):
    tubes = pd.read_csv(TUBES)
    tubes["od600"] = range(len(tubes))
    corr = read_corrections(_write(tmp_path, [_entry()]))["corrections"]
    out, changes = correct_tube_table(tubes, corr)

    a0 = tubes.set_index("sample").loc[A]
    b0 = tubes.set_index("sample").loc[B]
    a1 = out.set_index("sample").loc[A]
    b1 = out.set_index("sample").loc[B]
    for c in DESIGN_COLUMNS:
        assert a1[c] == b0[c]
        assert b1[c] == a0[c]
    # Measurements stay with the physical sample.
    assert a1["od600"] == a0["od600"]
    assert b1["od600"] == b0["od600"]
    assert a1["library"] == a0["library"]
    assert a1["correction_id"] == "c1" and b1["correction_id"] == "c1"
    # Untouched tubes keep empty correction ids; order is kept.
    assert list(out["sample"]) == list(tubes["sample"])
    assert (out.loc[~out["sample"].isin([A, B]), "correction_id"] == "").all()
    # Only the column that differed is recorded.
    assert set(changes["column"]) == {"titrant_conc"}
    assert len(changes) == 2


def test_relabel(tmp_path):
    out, changes = _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                            set={"t_sel": 125})])
    row = out.set_index("sample").loc[KAN]
    assert row["t_sel"] == 125
    assert pd.api.types.is_integer_dtype(out["t_sel"])
    assert changes.iloc[0][["sample", "column", "original", "corrected"]
                           ].tolist() == [KAN, "t_sel", 120, 125]


def test_exclude(tmp_path):
    out, changes = _apply(tmp_path, [_entry(action="exclude", tubes=[A, KAN])])
    assert len(out) == len(pd.read_csv(TUBES)) - 2
    assert not out["sample"].isin([A, KAN]).any()
    assert list(changes["action"]) == ["exclude", "exclude"]


def test_rotation_by_three_relabels(tmp_path):
    # kanR selection, t 60: rotate 0 mM -> 1 mM labels through a third tube's
    # time, which would collide one relabel at a time but not all together.
    t0 = "kanR_kanRpkan_0mM_t60"
    t1 = "kanR_kanRpkan_1mM_t60"
    t2 = "kanR_kanRpkan_0mM_t120"
    entries = [
        _entry(id="r1", action="relabel", tubes=[t0], set={"titrant_conc": 1.0}),
        _entry(id="r2", action="relabel", tubes=[t1],
               set={"titrant_conc": 0.0, "t_sel": 120}),
        _entry(id="r3", action="relabel", tubes=[t2], set={"t_sel": 60}),
    ]
    out, _ = _apply(tmp_path, entries)
    out = out.set_index("sample")
    assert out.loc[t0, "titrant_conc"] == 1.0
    assert (out.loc[t1, "titrant_conc"], out.loc[t1, "t_sel"]) == (0.0, 120)
    assert out.loc[t2, "t_sel"] == 60


def test_example_corrections_file(tmp_path):
    out = apply_corrections(TUBES, CORRECTIONS,
                            out_prefix=str(tmp_path / "fixed"))
    df = pd.read_csv(out)
    assert len(df) == len(pd.read_csv(TUBES)) - 1
    changes = pd.read_csv(tmp_path / "fixed_changes.csv")
    assert set(changes["action"]) == {"swap", "relabel", "exclude"}


# ---------------------------------------------------------------------------
# outputs, provenance and inputs
# ---------------------------------------------------------------------------

def test_inputs_untouched_and_provenance(tmp_path):
    evidence = tmp_path / "qc.csv"
    evidence.write_text("tube,apparent\nA,0.9\n")
    corr = _write(tmp_path, [_entry(evidence=["qc.csv", "a note"])])
    before = (_sha(TUBES), _sha(corr), os.path.getmtime(TUBES))

    prefix = str(tmp_path / "out" / "fixed")
    correct_tubes(TUBES, corr, out_prefix=prefix)

    assert (_sha(TUBES), _sha(corr), os.path.getmtime(TUBES)) == before
    prov = json.load(open(f"{prefix}_provenance.json"))
    assert prov["inputs"]["tube_table"]["sha256"] == before[0]
    assert prov["inputs"]["corrections_file"]["sha256"] == before[1]
    assert prov["outputs"]["tube_table"]["sha256"] == _sha(f"{prefix}.csv")
    assert prov["design_columns"] == list(DESIGN_COLUMNS)
    c = prov["corrections"][0]
    assert c["id"] == "c1" and c["reason"] == "QC"
    assert c["evidence"][0]["sha256"] == _sha(evidence)
    assert "sha256" not in c["evidence"][1]
    assert "tfscreen_version" in prov


def test_refuses_to_overwrite_input(tmp_path):
    tubes = tmp_path / "t.csv"
    pd.read_csv(TUBES).to_csv(tubes, index=False)
    corr = _write(tmp_path, [_entry()])
    with pytest.raises(ValueError, match="overwrite"):
        apply_corrections(str(tubes), corr, out_prefix=str(tmp_path / "t"))


def test_corrected_table_through_process_counts(tmp_path):
    od = os.path.join(RAW, "od600.csv")
    counts = os.path.join(RAW, "counts")
    kw = dict(od600_file=od, od600_calibration_file=CALIBRATION,
              tube_volume_mL=5.0, verbose=False)

    fixed = apply_corrections(TUBES, _write(tmp_path, [_entry()]),
                              out_prefix=str(tmp_path / "fixed"))
    g0 = pd.read_csv(process_counts(TUBES, counts,
                                    out_prefix=str(tmp_path / "g0"), **kw))
    g1 = pd.read_csv(process_counts(fixed, counts,
                                    out_prefix=str(tmp_path / "g1"), **kw))

    key = ["sample", "genotype"]
    m = g0.merge(g1, on=key, suffixes=("_0", "_1"))
    assert len(m) == len(g0) == len(g1)
    # Each sample keeps its own reads and tube total.
    for c in ("counts", "sample_reads", "sample_cfu", "sample_cfu_std"):
        assert (m[f"{c}_0"] == m[f"{c}_1"]).all()
    # The swapped samples carry each other's labels.
    lab = m.groupby("sample")[["titrant_conc_0", "titrant_conc_1"]].first()
    assert lab.loc[A, "titrant_conc_1"] == lab.loc[B, "titrant_conc_0"] == 1.0
    assert lab.loc[B, "titrant_conc_1"] == lab.loc[A, "titrant_conc_0"] == 0.0
    others = lab.drop(index=[A, B])
    assert (others["titrant_conc_0"] == others["titrant_conc_1"]).all()
    assert set(g1.loc[g1["sample"].isin([A, B]), "correction_id"]) == {"c1"}


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_unknown_tube(tmp_path):
    with pytest.raises(ValueError, match="no tube named"):
        _apply(tmp_path, [_entry(tubes=[A, "nope"])])


def test_tube_in_two_corrections(tmp_path):
    with pytest.raises(ValueError, match="also named by correction 'c1'"):
        _apply(tmp_path, [_entry(),
                          _entry(id="c2", action="exclude", tubes=[B])])


def test_swap_across_libraries(tmp_path):
    with pytest.raises(ValueError, match="different libraries"):
        _apply(tmp_path, [_entry(tubes=[A, KAN])])


def test_swap_that_changes_nothing(tmp_path):
    tubes = pd.read_csv(TUBES)
    twin = tubes[tubes["sample"] == A].assign(sample="twin")
    tubes = pd.concat([tubes, twin], ignore_index=True)
    corr = read_corrections(_write(tmp_path, [_entry(tubes=[A, "twin"])]))
    with pytest.raises(ValueError, match="changes nothing"):
        correct_tube_table(tubes, corr["corrections"])


@pytest.mark.parametrize("key", ["reason", "by", "date", "id", "action"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_required_field(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"'{key}' is required"):
        read_corrections(_write(tmp_path, [_entry(**{key: value})]))


def test_missing_reason(tmp_path):
    e = _entry()
    del e["reason"]
    with pytest.raises(ValueError, match="'reason' is required"):
        read_corrections(_write(tmp_path, [e]))


@pytest.mark.parametrize("entry, match", [
    (dict(action="flip"), "unknown action"),
    (dict(tubes=[A]), "swap takes exactly 2"),
    (dict(action="relabel", tubes=[A, B], set={"t_sel": 1}),
     "relabel takes exactly 1"),
    (dict(action="relabel", tubes=[A]), "needs a non-empty 'set'"),
    (dict(action="relabel", tubes=[A], set={"library": "kanR"}),
     "only design columns"),
    (dict(action="relabel", tubes=[A], set={"od600": 0.3}),
     "only design columns"),
    (dict(set={"t_sel": 1}), "'set' is only for relabel"),
    (dict(tubes=[A, A]), "listed twice"),
    (dict(tubes=[]), "at least one tube"),
    (dict(colour="red"), "unknown key"),
])
def test_bad_entries(tmp_path, entry, match):
    with pytest.raises(ValueError, match=match):
        read_corrections(_write(tmp_path, [_entry(**entry)]))


def test_duplicate_id(tmp_path):
    with pytest.raises(ValueError, match="duplicate id"):
        read_corrections(_write(tmp_path, [
            _entry(), _entry(action="exclude", tubes=[KAN])]))


def test_bad_file_structure(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("corrections: []\n")
    with pytest.raises(ValueError, match="non-empty list"):
        read_corrections(str(p))
    p.write_text("fixes: []\n")
    with pytest.raises(ValueError, match="'corrections' list"):
        read_corrections(str(p))
    p.write_text(textwrap.dedent("""\
        corrections: [{id: a, action: exclude, tubes: [x], reason: r,
                       by: b, date: 2026-10-15}]
        notes: hi
        """))
    with pytest.raises(ValueError, match="unknown top-level"):
        read_corrections(str(p))


def test_yaml_date_becomes_iso_string(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("corrections: [{id: a, action: exclude, tubes: [x], "
                 "reason: r, by: b, date: 2026-10-15}]\n")
    assert read_corrections(str(p))["corrections"][0]["date"] == "2026-10-15"


def test_relabel_that_changes_nothing(tmp_path):
    with pytest.raises(ValueError, match="changes nothing"):
        _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                 set={"t_sel": 120})])


def test_relabel_type_checked(tmp_path):
    with pytest.raises(ValueError, match="numeric"):
        _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                 set={"t_sel": "late"})])
    with pytest.raises(ValueError, match="integers"):
        _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                 set={"t_sel": 12.5})])


def test_relabel_column_absent_from_table(tmp_path):
    tubes = pd.read_csv(TUBES).drop(columns="t_pre")
    corr = read_corrections(_write(tmp_path, [
        _entry(action="relabel", tubes=[KAN], set={"t_pre": 20})]))
    with pytest.raises(ValueError, match="no column"):
        correct_tube_table(tubes, corr["corrections"])


def test_relabel_onto_another_tube(tmp_path):
    with pytest.raises(ValueError, match="share a library and design"):
        _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                 set={"t_sel": 60})])


def test_refuses_already_corrected_table(tmp_path):
    tubes = pd.read_csv(TUBES).assign(correction_id="")
    corr = read_corrections(_write(tmp_path, [_entry()]))
    with pytest.raises(ValueError, match="corrected before"):
        correct_tube_table(tubes, corr["corrections"])


def test_refuses_duplicate_or_missing_sample(tmp_path):
    corr = read_corrections(_write(tmp_path, [_entry()]))["corrections"]
    tubes = pd.read_csv(TUBES)
    with pytest.raises(ValueError, match="Duplicate tube name"):
        correct_tube_table(pd.concat([tubes, tubes.iloc[:1]]), corr)
    with pytest.raises(ValueError, match="no 'sample' column"):
        correct_tube_table(tubes.drop(columns="sample"), corr)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_cli_module_runs_as_script(monkeypatch):
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["tfs-correct-tubes", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("tfscreen.process_raw.scripts.correct_tubes_cli",
                         run_name="__main__")
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# edge cases of reading and applying
# ---------------------------------------------------------------------------

def test_correction_that_is_not_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="correction 1: each correction "
                                         "must be a mapping"):
        read_corrections(_write(tmp_path, ["swap A and B"]))


def test_single_tube_may_be_a_bare_string(tmp_path):
    corr = read_corrections(_write(tmp_path, [
        _entry(action="exclude", tubes=KAN)]))["corrections"]
    assert corr[0]["tubes"] == [KAN]
    df, changes = correct_tube_table(pd.read_csv(TUBES), corr)
    assert KAN not in set(df["sample"])
    assert changes["action"].tolist() == ["exclude"]


def test_empty_tube_name(tmp_path):
    with pytest.raises(ValueError, match="empty tube name"):
        read_corrections(_write(tmp_path, [
            _entry(action="exclude", tubes=[KAN, "  "])]))


def test_swap_with_a_column_missing_in_both_tubes(tmp_path):
    tubes = pd.read_csv(TUBES)
    tubes.loc[tubes["sample"].isin([A, B]), "titrant_name"] = float("nan")
    corr = read_corrections(_write(tmp_path, [_entry()]))["corrections"]
    df, changes = correct_tube_table(tubes, corr)
    # Both NaN counts as the same value: only titrant_conc changes.
    assert set(changes["column"]) == {"titrant_conc"}
    df = df.set_index("sample")
    orig = tubes.set_index("sample")
    assert df.at[A, "titrant_conc"] == orig.at[B, "titrant_conc"]
    assert df.loc[[A, B], "titrant_name"].isna().all()


def test_relabel_text_column(tmp_path):
    df, changes = _apply(tmp_path, [_entry(action="relabel", tubes=[KAN],
                                           set={"condition_sel": "kanR+new"})])
    assert df.set_index("sample").at[KAN, "condition_sel"] == "kanR+new"
    assert changes["corrected"].tolist() == ["kanR+new"]


def test_tube_table_indexed_by_sample(tmp_path, monkeypatch):
    import tfscreen.process_raw.tube_corrections as TC
    corr = _write(tmp_path, [_entry()])
    plain = str(tmp_path / "plain")
    apply_corrections(TUBES, corr, out_prefix=plain)

    real = TC.read_dataframe
    monkeypatch.setattr(TC, "read_dataframe",
                        lambda src: real(src).set_index("sample"))
    indexed = str(tmp_path / "indexed")
    apply_corrections(TUBES, corr, out_prefix=indexed)
    pd.testing.assert_frame_equal(pd.read_csv(f"{indexed}.csv"),
                                  pd.read_csv(f"{plain}.csv"))
