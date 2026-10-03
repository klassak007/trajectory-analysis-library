"""Inspect the independent wheels and source archives required by Contract 135."""

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

LICENSES = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}


def _wheel(path, package, forbidden):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert any(name.startswith(package + "/") for name in names)
        assert not any(name.startswith(forbidden + "/") for name in names)
        assert LICENSES <= {
            Path(name).name for name in names if ".dist-info/licenses/" in name
        }
        modules = [name for name in names if name.endswith(".py")]
        assert all(name.startswith(package + "/") for name in modules)
        assert not any(name.startswith(("tal/geo/", "tal/astro/")) for name in names)


def _sdist(path, package, forbidden):
    with tarfile.open(path) as archive:
        names = [name.split("/", 1)[-1] for name in archive.getnames()]
        assert any(
            name.startswith((package + "/", "src/" + package + "/")) for name in names
        )
        assert not any(
            name.startswith((forbidden + "/", "src/" + forbidden + "/"))
            for name in names
        )
        assert LICENSES <= set(names)
        if package == "tal":
            assert not any(name.startswith("extensions/") for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("extensions", type=Path)
    args = parser.parse_args()
    checked = []
    for directory, package, forbidden in (
        (args.base, "tal", "tal_extensions"),
        (args.extensions, "tal_extensions", "tal"),
    ):
        wheels, sources = (
            list(directory.glob("*.whl")),
            list(directory.glob("*.tar.gz")),
        )
        assert len(wheels) == len(sources) == 1
        _wheel(wheels[0], package, forbidden)
        _sdist(sources[0], package, forbidden)
        checked.extend(str(path) for path in (*wheels, *sources))
    print(json.dumps({"artifacts": checked, "result": "passed"}))


if __name__ == "__main__":
    main()
