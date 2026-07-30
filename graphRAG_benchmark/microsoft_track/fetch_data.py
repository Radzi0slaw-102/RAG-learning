# fetch and unpack the Microsoft graphrag-benchmarking-datasets repo
from __future__ import annotations

import gzip
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/microsoft/graphrag-benchmarking-datasets.git"
DATA_DIR = Path(__file__).parent / "data"
REPO_DIR = DATA_DIR / "_repo"

HOTPOT_SAMPLE_SIZE = 30


def clone_repo() -> Path:
    if REPO_DIR.exists():
        return REPO_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR


def unpack_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "rb") as f:
        magic = f.read(2)

    if magic == b"PK":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        return

    tmp_tar = dest_dir / "_tmp.tar"
    with gzip.open(archive_path, "rb") as src, open(tmp_tar, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with tarfile.open(tmp_tar) as tar:
        tar.extractall(dest_dir)
    tmp_tar.unlink()


def prepare_kevin_scott(repo_dir: Path) -> tuple[Path, Path]:
    archive = repo_dir / "data" / "Kevin Scott Podcast Transcripts Input Text.zip"
    dest = DATA_DIR / "kevin_scott" / "input"
    if not dest.exists():
        unpack_archive(archive, DATA_DIR / "kevin_scott")
    questions_csv = repo_dir / "data" / "Kevin Scott Questions.csv"
    return dest, questions_csv


def prepare_hotpotqa(repo_dir: Path, sample_size: int = HOTPOT_SAMPLE_SIZE) -> tuple[Path, Path]:
    archive = repo_dir / "data" / "HotPotQA Filtered Input Text.zip"
    full_dir = DATA_DIR / "hotpotqa_full"
    if not full_dir.exists():
        unpack_archive(archive, full_dir)

    sample_dir = DATA_DIR / "hotpotqa_sample" / "input"
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True, exist_ok=True)
        all_files = sorted((full_dir / "input").glob("*.txt"))
        for f in all_files[:sample_size]:
            shutil.copy(f, sample_dir / f.name)

    questions_csv = repo_dir / "data" / "HotPotQA Filtered Questions.csv"
    return sample_dir, questions_csv


def main() -> None:
    repo_dir = clone_repo()
    kevin_input, kevin_csv = prepare_kevin_scott(repo_dir)
    hotpot_input, hotpot_csv = prepare_hotpotqa(repo_dir)
    print(f"Kevin Scott input dir: {kevin_input}")
    print(f"Kevin Scott questions: {kevin_csv}")
    print(f"HotPotQA sample input dir: {hotpot_input} ({len(list(hotpot_input.glob('*.txt')))} docs)")
    print(f"HotPotQA questions: {hotpot_csv}")


if __name__ == "__main__":
    main()