"""Preview, check, or explicitly run a historical lls48 B224 stage."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from curconmix_release.spatial_training import build_command, main  # noqa: E402,F401


if __name__ == "__main__":
    main("lls48", Path(__file__).resolve().parent)
