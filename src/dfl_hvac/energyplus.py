"""EnergyPlus process wrapper and output utilities."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EnergyPlusRun:
    output_directory: Path
    stdout: str
    stderr: str


class EnergyPlusCLI:
    def __init__(self, executable: str = "energyplus"):
        self.executable = executable

    def executable_path(self) -> str:
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise FileNotFoundError(
                f"EnergyPlus executable '{self.executable}' was not found on PATH"
            )
        return resolved

    def version(self) -> str:
        result = subprocess.run(
            [self.executable_path(), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr).strip()

    def run(
        self,
        *,
        model_path,
        weather_path,
        output_directory,
        readvars: bool = True,
        extra_args=(),
    ) -> EnergyPlusRun:
        model = Path(model_path).expanduser().resolve()
        weather = Path(weather_path).expanduser().resolve()
        output = Path(output_directory).expanduser().resolve()

        if not model.is_file():
            raise FileNotFoundError(model)
        if not weather.is_file():
            raise FileNotFoundError(weather)

        output.mkdir(parents=True, exist_ok=True)
        command = [
            self.executable_path(),
            "-w",
            str(weather),
            "-d",
            str(output),
        ]
        if readvars:
            command.append("-r")
        command.extend(str(arg) for arg in extra_args)
        command.append(str(model))

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "EnergyPlus failed with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        return EnergyPlusRun(
            output_directory=output,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def read_energyplus_csv(path, *, required_columns=()) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing EnergyPlus output columns: {missing}")
    return frame
