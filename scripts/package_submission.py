"""Package a clean committed snapshot without data downloads or fitted artifacts."""
import argparse
import datetime
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Milestone1_Draft.zip")
    args = parser.parse_args()
    if git("status", "--porcelain").strip():
        raise SystemExit("Commit the final changes before packaging; the working tree must be clean.")
    commit = git("rev-parse", "HEAD").decode().strip()
    run = json.loads((ROOT / "results/run_summary.json").read_text())
    verification = json.loads((ROOT / "results/verification.json").read_text())
    if run["final_test_evaluated"] or verification["final_test_evaluated"]:
        raise SystemExit("This Milestone 1 package expects the final test evaluation to remain reserved.")
    paths = git("ls-files", "-z").decode().strip("\0").split("\0")
    entries = {}
    for relative in paths:
        path = Path(relative)
        if path.parts[0] in (".git", ".venv", "artifacts") and path.name != ".gitkeep":
            continue
        if path.parts[:2] == ("data", "raw") and path.name != ".gitkeep":
            continue
        if path.suffix.lower() in (".joblib", ".pkl", ".pickle", ".pyc", ".zip"):
            continue
        if any(part in ("__pycache__", ".git", ".venv") for part in path.parts) or path.name.startswith(".env"):
            continue
        # Read exact committed bytes, independent of checkout line endings.
        content = git("show", f"HEAD:{relative}")
        if relative == "README.md":
            text = content.decode("utf-8")
            if text.count("__SUBMITTED_COMMIT_SHA__") != 1:
                raise SystemExit("Expected one README commit placeholder.")
            content = text.replace("__SUBMITTED_COMMIT_SHA__", commit).encode("utf-8")
        entries[relative] = content
    metadata = {
        "repository": "https://github.com/Omar-Medhat1/CSE382-Hotel-Booking",
        "submitted_commit_sha": commit,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "technical_draft_team_details_pending",
        "readme_metadata_substitution": "The archive README replaces its commit placeholder with the source snapshot SHA; the checkout is unchanged.",
        "source_data_sha256": run["source"]["sha256"],
        "final_test_evaluated": False,
        "files_sha256": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(entries.items())},
    }
    entries["SUBMISSION.json"] = json.dumps(metadata, indent=2).encode("utf-8")
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, content in sorted(entries.items()):
            archive.writestr(f"{ROOT.name}/{path}", content)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert not any(".joblib" in path or "/.venv/" in path or path.endswith("/H2.csv") for path in archive.namelist())
        assert commit in archive.read(f"{ROOT.name}/README.md").decode()
    print(json.dumps({"zip": str(destination), "source_commit": commit, "files": len(entries),
                      "bytes": destination.stat().st_size, "status": metadata["status"]}, indent=2))


if __name__ == "__main__":
    main()
