"""Rebuild wheels from each independent source archive, outside the checkout."""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for name in ("base", "extensions"):
        sources = list((args.artifacts / name).glob("*.tar.gz"))
        assert len(sources) == 1
        with tempfile.TemporaryDirectory(prefix="tal-sdist-") as directory:
            with tarfile.open(sources[0]) as archive:
                archive.extractall(directory, filter="data")
            roots = list(Path(directory).iterdir())
            assert len(roots) == 1
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str((args.output / name).resolve()),
                    str(roots[0]),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
