import sys
from pathlib import Path


def test_imports():
    # Ensure the repository is importable for basic sanity
    src = Path(__file__).resolve().parents[1] / "source"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import stable_yield_lab  # noqa: F401

