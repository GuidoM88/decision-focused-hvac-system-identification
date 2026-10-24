"""Validate the local EnergyPlus installation."""

from __future__ import annotations

import argparse

from dfl_hvac.energyplus import EnergyPlusCLI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", default="energyplus")
    args = parser.parse_args()

    cli = EnergyPlusCLI(args.executable)
    print(cli.executable_path())
    print(cli.version())


if __name__ == "__main__":
    main()
