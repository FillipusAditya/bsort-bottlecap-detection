import os
from pathlib import Path
import pytest
from scripts.relabel import LabelRelabeler


# Helper: Create dummy label files
def create_dummy_labels(tmp_path, files_content):
    """
    Create dummy YOLO label files inside a temporary label directory.

    files_content format:
    {
        "raw_b2_1.txt": ["0 0.5 0.5 0.2 0.2"],
        "raw_b5_1.txt": ["1 0.4 0.4 0.3 0.3"],
        ...
    }
    """
    label_dir = tmp_path / "labels"
    label_dir.mkdir()

    for fname, lines in files_content.items():
        file_path = label_dir / fname
        with open(file_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return label_dir


# TEST 1 — extract_code()
def test_extract_code():
    relabeler = LabelRelabeler("dummy", "dummy_out", {})

    assert relabeler.extract_code("img_b2_123.txt") == "b2"
    assert relabeler.extract_code("abc_b5_.txt") == "b5"
    assert relabeler.extract_code("wrongpattern.txt") is None
    assert relabeler.extract_code("test_b9_1.txt") == "b9"


# TEST 2 — relabel a single file (normal case)
def test_relabel_single_file(tmp_path):
    # Create dummy label file
    label_dir = create_dummy_labels(tmp_path, {
        "cap_b4_1.txt": ["0 0.3 0.3 0.1 0.1"]
    })

    out_dir = tmp_path / "out"
    color_map = {"b4": 0}

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.relabel_file("cap_b4_1.txt")

    # Output file should exist
    out_file = out_dir / "cap_b4_1.txt"
    assert out_file.exists()

    # Class id should be updated
    with open(out_file) as f:
        line = f.readline().strip()
    assert line.startswith("0 ")


# TEST 3 — relabel multiple files
def test_relabel_multiple(tmp_path):
    label_dir = create_dummy_labels(tmp_path, {
        "cap_b2_1.txt": ["0 0.1 0.1 0.2 0.2"],
        "cap_b5_9.txt": ["1 0.3 0.3 0.1 0.1"],
    })

    out_dir = tmp_path / "out"
    color_map = {"b2": 2, "b5": 1}

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.run()

    # Output files must exist
    assert (out_dir / "cap_b2_1.txt").exists()
    assert (out_dir / "cap_b5_9.txt").exists()

    # Correct class mapping
    with open(out_dir / "cap_b2_1.txt") as f:
        assert f.readline().split()[0] == "2"

    with open(out_dir / "cap_b5_9.txt") as f:
        assert f.readline().split()[0] == "1"


# TEST 4 — skip file if b-code is missing
def test_skip_missing_bcode(tmp_path, capsys):
    label_dir = create_dummy_labels(tmp_path, {
        "wrongname.txt": ["0 0.2 0.3 0.5 0.5"]
    })

    out_dir = tmp_path / "out"
    color_map = {"b2": 2}

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.relabel_file("wrongname.txt")

    # Output should not be created
    assert not (out_dir / "wrongname.txt").exists()

    captured_output = capsys.readouterr().out
    assert "missing b-code" in captured_output  # Expect English message


# TEST 5 — skip file if no mapping is defined for the b-code
def test_skip_no_mapping(tmp_path, capsys):
    label_dir = create_dummy_labels(tmp_path, {
        "cap_b7_1.txt": ["0 0.2 0.3 0.2 0.2"]
    })

    out_dir = tmp_path / "out"
    color_map = {"b4": 0}  # No mapping for b7

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.relabel_file("cap_b7_1.txt")

    assert not (out_dir / "cap_b7_1.txt").exists()

    captured_output = capsys.readouterr().out
    assert "no mapping for b7" in captured_output


# TEST 6 — number of lines must remain unchanged after relabel
def test_line_count_preserved(tmp_path):
    label_dir = create_dummy_labels(tmp_path, {
        "cap_b3_1.txt": [
            "0 0.1 0.2 0.3 0.4",
            "1 0.5 0.6 0.3 0.2",
            "2 0.7 0.8 0.1 0.1",
        ]
    })

    out_dir = tmp_path / "out"
    color_map = {"b3": 2}

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.relabel_file("cap_b3_1.txt")

    out_file = out_dir / "cap_b3_1.txt"
    assert out_file.exists()

    with open(out_file) as f:
        lines = f.readlines()

    assert len(lines) == 3
    assert all(line.startswith("2 ") for line in lines)


# TEST 7 — run(): process all files in folder
def test_run_process_all(tmp_path):
    label_dir = create_dummy_labels(tmp_path, {
        "cap_b2_1.txt": ["0 0.1 0.1 0.2 0.2"],
        "cap_b3_1.txt": ["1 0.1 0.1 0.2 0.2"],
    })

    out_dir = tmp_path / "out"
    color_map = {"b2": 9, "b3": 8}

    relabeler = LabelRelabeler(label_dir, out_dir, color_map)
    relabeler.run()

    # Both files must exist
    assert (out_dir / "cap_b2_1.txt").exists()
    assert (out_dir / "cap_b3_1.txt").exists()

    # Correct mapping applied
    with open(out_dir / "cap_b2_1.txt") as f:
        assert f.readline().split()[0] == "9"

    with open(out_dir / "cap_b3_1.txt") as f:
        assert f.readline().split()[0] == "8"
