import numpy as np
import pandas as pd
import pytest

from mindtext.cli import main
from mindtext.errors import DataError
from mindtext.ingest import build_acoustic_features, infer_label, prepare_reddit

LEGACY = {"ani": "anxiety", "dep": "depression", "lon": "lonely", "mhj": "mentalhealth", "swj": "suicidewatch"}


def legacy_label(filename):  # the original notebook's rule
    return LEGACY.get(filename[:3].lower(), "unknown")


@pytest.mark.parametrize("name", ["depnov20.csv", "lonenov20.csv", "SWjan20.csv", "Mhjan20.csv", "anxijan20.csv",
                                  "swnov20.csv", "mhov20.csv", "anxinov20.csv", "Swaug20 copy.csv", "Mheaug20.csv", "depjan20 .csv"])
def test_every_observed_file_name_gets_a_label(name):
    assert infer_label(name) is not None


def test_new_rules_recover_files_the_legacy_rule_discarded():
    recovered = [n for n in ["swnov20.csv", "mhov20.csv", "anxinov20.csv", "Swaug20 copy.csv", "Mheaug20.csv"]
                 if legacy_label(n) == "unknown" and infer_label(n) is not None]
    assert len(recovered) == 5


def test_prepare_reddit(tmp_path):
    raw = tmp_path / "raw" / "2020" / "Nov 20"
    raw.mkdir(parents=True)
    for fname, n in [("depnov20.csv", 3), ("swnov20.csv", 2), ("Swaug20 copy.csv", 2), ("readme_notes.csv", 1)]:
        pd.DataFrame({"selftext": [f"post {fname} {i}" for i in range(n)], "id": [f"i{i}" for i in range(n)],
                      "author": ["alice"] * n, "created_utc": [1600000000 + i for i in range(n)]}).to_csv(raw / fname, index=False)
    pd.DataFrame({"title": ["x"]}).to_csv(raw / "lonenov20.csv", index=False)  # no selftext column
    out = tmp_path / "out" / "all.csv"
    s = prepare_reddit(tmp_path / "raw", out, hash_salt="s")
    df = pd.read_csv(out)
    assert s["per_label"] == {"depression": 3, "suicidewatch": 4} and len(df) == 7
    assert s["unmatched_files"] == ["2020/Nov 20/readme_notes.csv"] and len(s["suspected_copies"]) == 1
    assert "alice" not in df["group"].iloc[0] and df["group"].nunique() == 1
    with pytest.raises(DataError, match="match no label rule"):
        prepare_reddit(tmp_path / "raw", tmp_path / "o2.csv", strict=True)
    with pytest.raises(DataError, match="not found"):
        prepare_reddit(tmp_path / "nowhere", out)


def _make_raw_edaic(root, labels_dir, ids_by_split, rng):
    labels_dir.mkdir(parents=True)
    for split, ids in ids_by_split.items():
        pd.DataFrame({"Participant_ID": ids, "PHQ_Binary": [i % 2 for i in ids], "PHQ_Score": 5, "Gender": "male"}).to_csv(labels_dir / f"{split}_split.csv", index=False)
        for pid in ids:
            d = root / f"{pid}_P"
            d.mkdir(parents=True)
            fr = pd.DataFrame(rng.normal(size=(200, 3)), columns=["Loudness_sma3", "F1frequency_sma3nz", "F2amplitudeLogRelF0_sma3nz"])
            fr.iloc[:50, 2] = -201.0
            fr.insert(0, "frameTime", np.arange(200) / 100)
            fr.insert(0, "name", "'unknown'")
            fr.to_csv(d / f"{pid}_OpenSMILE2.3.0_egemaps.csv", sep=";", index=False)
            fr.to_csv(d / f"{pid}_BoAW_openSMILE_2.3.0_eGeMAPS.csv", sep=";", index=False)  # must be ignored


def test_build_acoustic_features(tmp_path):
    rng = np.random.default_rng(0)
    raw, lab = tmp_path / "raw", tmp_path / "labels"
    _make_raw_edaic(raw, lab, {"train": [300, 301, 302], "dev": [310], "test": [320]}, rng)
    (raw / "301_P" / "301_OpenSMILE2.3.0_egemaps.csv").unlink()
    out = tmp_path / "feat.csv"
    s = build_acoustic_features(raw, lab, out)
    df = pd.read_csv(out)
    assert s["missing_participants"] == [301] and len(df) == 4
    assert list(df.columns[:3]) == ["participant_id", "split", "label"] and "PHQ_Score" not in df.columns and "Gender" not in df.columns
    assert s["n_features"] == 3 * 7 and df.shape[1] == 3 + 21
    # sentinel handling: the -201 frames must not drag the mean of the third feature to about -50
    assert df["eg_F2amplitudeLogRelF0_sma3nz_mean"].abs().max() < 1
    build_acoustic_features(raw, lab, tmp_path / "f2.csv", sentinels=(), nan_fill=0.0)
    assert pd.read_csv(tmp_path / "f2.csv")["eg_F2amplitudeLogRelF0_sma3nz_mean"].min() < -30


def test_build_acoustic_features_errors_and_cli(tmp_path):
    with pytest.raises(DataError, match="not found"):
        build_acoustic_features(tmp_path / "x", tmp_path / "y", tmp_path / "o.csv")
    assert main(["build-acoustic-features", "--raw-dir", str(tmp_path / "x"), "--labels-dir", str(tmp_path / "y"), "--out", str(tmp_path / "o.csv")]) == 2
    assert main(["prepare-reddit", "--raw-dir", str(tmp_path / "x"), "--out", str(tmp_path / "o.csv")]) == 2
