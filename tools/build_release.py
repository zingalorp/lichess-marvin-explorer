"""Build the self-contained Windows x64 ZIP using a full Python 3.13 with pip.

Run from a source checkout:
  python tools/build_release.py --output dist
Use --archive-only PACKAGE_FOLDER to archive an already assembled payload.
Existing payloads/archives are never overwritten. Build artifacts are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.13.15"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
PYTHON_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
RUNTIME_FILES = (
    "extension-id.txt", "install.ps1", "setup.ps1", "Setup.cmd", "Uninstall.cmd",
    "README.md", "START-HERE.txt", "THIRD-PARTY-NOTICES.txt",
    "host/native_host.py", "host/explorer.py", "host/temporal.py",
    "host/temporal-profile.json", "host/requirements.txt", "host/check_installation.py",
    "host/marvin/__init__.py", "host/marvin/encoding.py", "host/marvin/inference.py",
    "host/marvin/timing.py",
    "weights/marvin_79m.onnx", "weights/marvin_79m.offset.npz",
    "weights/marvin_23m.onnx", "weights/marvin_23m.offset.npz",
    "tools/build_release.py", "tools/bundle-requirements.txt",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payload_files(payload):
    for path in sorted(payload.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(payload)
        if any(p in {"__pycache__", ".git", "tests", "test-results", "bin"} for p in rel.parts):
            continue
        if path.suffix in {".pyc", ".log"} or rel.as_posix() in {
            "host-config.json", "host/run_host.cmd", "com.marvin.lichess_explorer.json",
            "SHA256SUMS.txt",
        }:
            continue
        yield path, rel.as_posix()


def archive(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    zip_path = output / f"{payload.name}.zip"
    if zip_path.exists():
        raise FileExistsError(f"Archive already exists: {zip_path}")
    files = list(payload_files(payload))
    (payload / "SHA256SUMS.txt").write_text(
        "".join(f"{digest(path)}  {rel}\n" for path, rel in files), encoding="utf-8"
    )
    files.append((payload / "SHA256SUMS.txt", "SHA256SUMS.txt"))
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path, rel in files:
            bundle.write(path, f"{payload.name}/{rel}")
    checksum = digest(zip_path)
    zip_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {zip_path.name}\n", encoding="ascii")
    print(f"Created {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(f"SHA-256: {checksum}", flush=True)
    return zip_path


def build(output):
    if sys.version_info[:2] != (3, 13) or sys.platform != "win32":
        raise RuntimeError("Build with 64-bit Windows Python 3.13 and pip.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("A 64-bit interpreter is required.")
    version = json.loads((ROOT / "extension/manifest.json").read_text(encoding="utf-8"))["version"]
    payload = output / f"Marvin-Opening-Explorer-{version}-Windows"
    payload.mkdir(parents=True, exist_ok=False)
    downloads = output / f"build-downloads-{version}"
    downloads.mkdir(exist_ok=True)
    for rel in RUNTIME_FILES:
        destination = payload / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    shutil.copytree(ROOT / "extension", payload / "extension")
    python_zip = downloads / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    if not python_zip.exists():
        urllib.request.urlretrieve(PYTHON_URL, python_zip)
    if digest(python_zip) != PYTHON_SHA256:
        raise RuntimeError("The official Python download checksum does not match.")
    with zipfile.ZipFile(python_zip) as bundle:
        bundle.extractall(payload / "runtime")
    (payload / "runtime/python313._pth").write_text(
        "python313.zip\n.\nLib/site-packages\n../host\n", encoding="utf-8"
    )
    subprocess.run([
        sys.executable, "-m", "pip", "wheel", "--index-url", "https://pypi.org/simple",
        "--wheel-dir", str(downloads), "-r", str(ROOT / "tools/bundle-requirements.txt"),
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(downloads),
        "--no-compile", "--target", str(payload / "runtime/Lib/site-packages"),
        "-r", str(ROOT / "tools/bundle-requirements.txt"),
    ], check=True)
    subprocess.run([
        str(payload / "runtime/python.exe"), str(payload / "host/check_installation.py"),
    ], check=True)
    return archive(payload, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--archive-only", type=Path)
    args = parser.parse_args()
    if args.archive_only:
        archive(args.archive_only.resolve(), args.output.resolve())
    else:
        build(args.output.resolve())
