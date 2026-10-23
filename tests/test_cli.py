import json
import subprocess
import sys

import pandas as pd


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "talent_graph", *map(str, args)],
        check=False,
        text=True,
        capture_output=True,
        cwd=cwd,
        timeout=90,
    )


def test_demo_validate_and_analyze_work_without_network(tmp_path):
    demo = tmp_path / "input"
    output = tmp_path / "results"
    r = run("demo", "--output", demo)
    assert r.returncode == 0, r.stderr
    assert (demo / "member_skillnote.csv").exists()
    validation = run("validate", "--data-dir", demo)
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout)["active_members"] == 48
    result = run(
        "analyze",
        "--data-dir",
        demo,
        "--reference-file",
        demo / "reference.csv",
        "--output",
        output,
        "--epochs",
        5,
    )
    assert result.returncode == 0, result.stderr
    members = pd.read_csv(output / "member_scores.csv", dtype={"member_code": str})
    assert len(members) == 48
    assert members.member_code.str.len().eq(4).all()
    assert (output / "skill_importance.csv").exists()
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["data_kind"] == "synthetic"
    assert metadata["python"].startswith("3.15")
    assert metadata["inputs"][0]["sha256"]
    assert (output / "report.md").exists()
    assert (
        run(
            "analyze",
            "--data-dir",
            demo,
            "--reference-file",
            demo / "reference.csv",
            "--output",
            output,
            "--epochs",
            2,
        ).returncode
        != 0
    )


def test_missing_input_and_reference_are_actionable(tmp_path):
    r = run("validate", "--data-dir", tmp_path)
    assert r.returncode == 2
    assert "必須CSV" in r.stderr
    assert "Traceback" not in r.stderr


def test_demo_refuses_to_replace_existing_folder(tmp_path):
    target = tmp_path / "input"
    target.mkdir()
    (target / "note.txt").write_text("keep")
    r = run("demo", "--output", target)
    assert r.returncode == 2
    assert (target / "note.txt").read_text() == "keep"


def test_output_cannot_be_written_inside_input(tmp_path):
    demo = tmp_path / "input"
    assert run("demo", "--output", demo).returncode == 0
    r = run(
        "analyze",
        "--data-dir",
        demo,
        "--reference-file",
        demo / "reference.csv",
        "--output",
        demo / "results",
        "--epochs",
        2,
    )
    assert r.returncode == 2
    assert not (demo / "results").exists()


def test_invalid_option_fails_before_output_creation(tmp_path):
    demo = tmp_path / "input"
    assert run("demo", "--output", demo).returncode == 0
    r = run(
        "analyze",
        "--data-dir",
        demo,
        "--reference",
        "0001",
        "--output",
        tmp_path / "results",
        "--epochs",
        0,
    )
    assert r.returncode == 2
    assert not (tmp_path / "results").exists()
