import os
from pathlib import Path
import pytest
from scripts.split_dataset import DatasetSplitter


# HELPER: create dummy dataset
def create_dataset(tmp_path, patterns):
    """
    Create dummy dataset based on a list of (filename) patterns.
    All files get a matching .txt label unless intentionally missing.
    """
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for fname in patterns:
        # Create dummy .jpg
        (img_dir / fname).write_text("img")

        # Create matching .txt unless explicitly missing
        if not fname.endswith("_nolabel.jpg"):
            (lbl_dir / fname.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.2 0.2")

    return img_dir, lbl_dir


# TEST 1 — clean_output_dirs(): folder cleanup + recreation
def test_clean_output_dirs(tmp_path):
    splitter = DatasetSplitter(
        img_dir="unused",
        lbl_dir="unused",
        out_img_train=str(tmp_path / "out/train/images"),
        out_img_val=str(tmp_path / "out/val/images"),
        out_lbl_train=str(tmp_path / "out/train/labels"),
        out_lbl_val=str(tmp_path / "out/val/labels"),
    )

    # Pre-create folders + files
    for d in [splitter.out_img_train, splitter.out_img_val,
              splitter.out_lbl_train, splitter.out_lbl_val]:
        os.makedirs(d, exist_ok=True)
        Path(d, "dummy.txt").write_text("x")

    splitter.clean_output_dirs()

    # Must exist and be empty
    for d in [splitter.out_img_train, splitter.out_img_val,
              splitter.out_lbl_train, splitter.out_lbl_val]:
        assert os.path.exists(d)
        assert len(os.listdir(d)) == 0


# TEST 2 — categorize_files(): bucket grouping
def test_categorize_files_basic(tmp_path):
    patterns = [
        "aaa_b2_1.jpg", "aaa_b2_2.jpg", "aaa_b2_3.jpg",
        "bbb_b3_1.jpg", "bbb_b3_2.jpg", "bbb_b3_3.jpg",
        "ccc_b4_1.jpg", "ccc_b4_2.jpg", "ccc_b4_3.jpg",
        "ddd_b5_1.jpg", "ddd_b5_2.jpg", "ddd_b5_3.jpg",
    ]

    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train="unused", out_img_val="unused",
        out_lbl_train="unused", out_lbl_val="unused"
    )

    splitter.categorize_files()

    assert len(splitter.buckets["other_b2"]) == 3
    assert len(splitter.buckets["other_b3"]) == 3
    assert len(splitter.buckets["lightblue_b4"]) == 3
    assert len(splitter.buckets["darkblue_b5"]) == 3


# TEST 3 — categorize_files(): wrong pattern
def test_categorize_files_wrong_pattern(tmp_path, capsys):
    patterns = ["abc123.jpg"]  # no `_bX_` pattern
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train="unused", out_img_val="unused",
        out_lbl_train="unused", out_lbl_val="unused",
    )

    splitter.categorize_files()
    output = capsys.readouterr().out

    assert "missing b-code pattern" in output
    assert all(len(v) == 0 for v in splitter.buckets.values())


# TEST 4 — categorize_files(): unknown b-code
def test_categorize_files_unknown_bcode(tmp_path, capsys):
    patterns = ["test_b9_x.jpg"]  # b9 is not defined
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train="unused", out_img_val="unused",
        out_lbl_train="unused", out_lbl_val="unused",
    )

    splitter.categorize_files()

    output = capsys.readouterr().out
    assert "Unknown b-code" in output


# TEST 5 — split_and_copy(): correct 2:1 split
def test_split_stratified_2_1(tmp_path):
    patterns = [
        "aaa_b2_1.jpg", "aaa_b2_2.jpg", "aaa_b2_3.jpg",
        "bbb_b3_1.jpg", "bbb_b3_2.jpg", "bbb_b3_3.jpg",
        "ccc_b4_1.jpg", "ccc_b4_2.jpg", "ccc_b4_3.jpg",
        "ddd_b5_1.jpg", "ddd_b5_2.jpg", "ddd_b5_3.jpg",
    ]
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    out_train_img = tmp_path / "train/img"
    out_val_img = tmp_path / "val/img"
    out_train_lbl = tmp_path / "train/lbl"
    out_val_lbl = tmp_path / "val/lbl"

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train=str(out_train_img), out_img_val=str(out_val_img),
        out_lbl_train=str(out_train_lbl), out_lbl_val=str(out_val_lbl),
    )

    splitter.clean_output_dirs()
    splitter.categorize_files()
    splitter.split_and_copy()

    assert len(list(out_train_img.glob("*.jpg"))) == 8
    assert len(list(out_val_img.glob("*.jpg"))) == 4
    assert len(list(out_train_lbl.glob("*.txt"))) == 8
    assert len(list(out_val_lbl.glob("*.txt"))) == 4


# TEST 6 — Image & label pairing
def test_image_label_pairing(tmp_path):
    patterns = ["aaa_b2_1.jpg", "aaa_b2_2.jpg", "aaa_b2_3.jpg"]
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    out_train_img = tmp_path / "train/img"
    out_train_lbl = tmp_path / "train/lbl"
    out_val_img = tmp_path / "val/img"
    out_val_lbl = tmp_path / "val/lbl"

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train=str(out_train_img), out_img_val=str(out_val_img),
        out_lbl_train=str(out_train_lbl), out_lbl_val=str(out_val_lbl),
    )

    splitter.run()

    for jpg in list(out_train_img.glob("*.jpg")) + list(out_val_img.glob("*.jpg")):
        txt = (out_train_lbl / jpg.name.replace(".jpg", ".txt"))
        txt2 = (out_val_lbl / jpg.name.replace(".jpg", ".txt"))

        assert txt.exists() or txt2.exists()


# TEST 7 — Missing label
def test_missing_label(tmp_path):
    patterns = ["aaa_b2_1_nolabel.jpg"]  # no .txt file
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    out_train_img = tmp_path / "out_train"
    out_val_img = tmp_path / "out_val"
    out_train_lbl = tmp_path / "out_train_lbl"
    out_val_lbl = tmp_path / "out_val_lbl"

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train=str(out_train_img), out_img_val=str(out_val_img),
        out_lbl_train=str(out_train_lbl), out_lbl_val=str(out_val_lbl),
    )

    splitter.clean_output_dirs()
    splitter.categorize_files()

    # Should not copy anything (safe behavior)
    splitter.split_and_copy()

    assert len(list(out_train_img.glob("*.jpg"))) == 0
    assert len(list(out_val_img.glob("*.jpg"))) == 0


# TEST 8 — Behavior when bucket has less than 3 files
def test_bucket_with_less_than_3_files(tmp_path, capsys):
    patterns = ["aaa_b4_1.jpg", "aaa_b4_2.jpg"]  # only 2 files
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    out_train_img = tmp_path / "train_img"
    out_val_img = tmp_path / "val_img"
    out_train_lbl = tmp_path / "train_lbl"
    out_val_lbl = tmp_path / "val_lbl"

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train=str(out_train_img), out_img_val=str(out_val_img),
        out_lbl_train=str(out_train_lbl), out_lbl_val=str(out_val_lbl),
    )

    splitter.clean_output_dirs()
    splitter.categorize_files()
    splitter.split_and_copy()

    output = capsys.readouterr().out
    assert "expected 3 files" in output  # warning should appear


# TEST 9 — Order determinism (sorted)
def test_order_is_sorted(tmp_path):
    patterns = ["aaa_b2_3.jpg", "aaa_b2_1.jpg", "aaa_b2_2.jpg"]
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train="unused", out_img_val="unused",
        out_lbl_train="unused", out_lbl_val="unused"
    )

    splitter.categorize_files()

    # Verify order is sorted
    assert splitter.buckets["other_b2"] == [
        "aaa_b2_1.jpg",
        "aaa_b2_2.jpg",
        "aaa_b2_3.jpg"
    ]


# TEST 10 — summary output
def test_summary_output(tmp_path, capsys):
    patterns = ["aaa_b2_1.jpg", "aaa_b2_2.jpg", "aaa_b2_3.jpg"]
    img_dir, lbl_dir = create_dataset(tmp_path, patterns)

    out_train_img = tmp_path / "train_img"
    out_val_img = tmp_path / "val_img"
    out_train_lbl = tmp_path / "train_lbl"
    out_val_lbl = tmp_path / "val_lbl"

    splitter = DatasetSplitter(
        img_dir=str(img_dir), lbl_dir=str(lbl_dir),
        out_img_train=str(out_train_img), out_img_val=str(out_val_img),
        out_lbl_train=str(out_train_lbl), out_lbl_val=str(out_val_lbl),
    )

    splitter.run()

    output = capsys.readouterr().out
    assert "Dataset Split Completed Successfully!" in output
    assert "Training Images" in output
    assert "Validation Images" in output
