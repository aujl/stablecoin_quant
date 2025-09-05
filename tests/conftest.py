import sys
from pathlib import Path


# Ensure the package is importable without installation when running tests locally
pkg_src = Path(__file__).resolve().parents[1] / "src"
if str(pkg_src) not in sys.path:
    sys.path.insert(0, str(pkg_src))
