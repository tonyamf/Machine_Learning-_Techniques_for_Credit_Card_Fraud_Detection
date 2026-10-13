import shutil
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

FIXTURE = Path(__file__).parent / "fixtures" / "creditcard_small.csv"
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fraud_module(monkeypatch):
    """Import the script as a module with plt.show disabled."""
    monkeypatch.syspath_prepend(str(ROOT))
    import matplotlib.pyplot as plt
    import fraud_detection_analysis as mod

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    return mod


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A temp cwd holding the synthetic creditcard.csv (the script reads it from cwd)."""
    shutil.copy(FIXTURE, tmp_path / "creditcard.csv")
    monkeypatch.chdir(tmp_path)
    return tmp_path
