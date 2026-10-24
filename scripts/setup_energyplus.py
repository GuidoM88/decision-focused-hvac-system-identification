"""Fetch the pinned Paper 2 case and optionally EnergyPlus 22.1 for Linux x86-64."""

import argparse
import hashlib
import json
import platform
import tarfile
from pathlib import Path
from urllib.request import urlopen

from dfl_hvac.public_data import AUTHOR_COMMIT

CASE_FILES = {
    "6zones_ASHRAE901_OfficeSmall_STD2022_Denver.idf": "4da5b2ef74443f60afdf8b0fa75473413d547136",
    "USA_CO_Denver.Intl.AP.725650_US.Normals.1991-2020.epw": "c77b5dc7299b3362b575aad671f08e5a871f975b",
    "USA_CO_Denver.Intl.AP.725650_US.Normals.2006-2020.epw": "c194189f97b01ce514e63c5f14f414edbd385b9e",
}
RELEASE = "EnergyPlus-22.1.0-ed759b17ee-Linux-Ubuntu20.04-x86_64"
ARCHIVE_SHA256 = "b8f0a446e1fb55791e113981a3524d70bf4046a49c8ae3b68c193677fd773a8b"


def fetch(path, url):
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        data = urlopen(url, timeout=180).read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-linux", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path("data/external"))
    args = parser.parse_args()
    sources = {}
    for name, expected in CASE_FILES.items():
        url = f"https://raw.githubusercontent.com/PSMRB/dfl_hvac_management/{AUTHOR_COMMIT}/data/SmallOffice/{name}"
        data = fetch(args.directory / "energyplus_case" / name, url)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != expected:
            raise ValueError(f"Checksum mismatch: {name}; remove the incomplete download and retry")
        sources[name] = dict(url=url, git_blob_sha1=blob, sha256=hashlib.sha256(data).hexdigest())
    (args.directory / "energyplus_case/sources.json").write_text(json.dumps(sources, indent=2))
    if args.install_linux:
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            raise RuntimeError("Automatic installation supports Linux x86-64 only; install 22.1 manually")
        root = args.directory / "energyplus"
        archive = root / f"{RELEASE}.tar.gz"
        url = f"https://github.com/NatLabRockies/EnergyPlus/releases/download/v22.1.0/{archive.name}"
        data = fetch(archive, url)
        if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Official EnergyPlus archive checksum mismatch")
        if not (root / RELEASE / "energyplus").exists():
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
        print(f"EnergyPlus installed at {(root / RELEASE).resolve()}")
    print("Case assets verified")


if __name__ == "__main__":
    main()
