"""Download the DOE Medium Office reference-building archive cited in Paper 1."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = (
    "https://www.energycodes.gov/sites/default/files/2023-10/"
    "ASHRAE901_OfficeMedium_STD2019.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/ASHRAE901_OfficeMedium_STD2019.zip"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0 dfl-hvac-reproduction"})
    try:
        with urlopen(request) as response, args.output.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
    except (HTTPError, URLError) as exc:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to download the DOE archive automatically. "
            f"Retrieve it manually from the official URL: {URL}"
        ) from exc

    print(f"path={args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
