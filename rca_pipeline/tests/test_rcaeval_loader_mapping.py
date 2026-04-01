"""Unit tests for RCAEvalLoader dataset mapping/detection."""

import tempfile
from pathlib import Path

from rca_pipeline.loaders.rcaeval_loader import RCAEvalLoader


def test_available_datasets_detects_re2_tt_layout():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data" / "RE2" / "RE2-TT").mkdir(parents=True)
        # Touch a file so `any(p.iterdir())` is true.
        (root / "data" / "RE2" / "RE2-TT" / "dummy.csv").write_text("timestamp\n0\n")

        loader = RCAEvalLoader(rcaeval_root=str(root))
        avail = loader.available_datasets()
        assert "trainticket" in avail


if __name__ == "__main__":
    test_available_datasets_detects_re2_tt_layout()
    print("ok: test_rcaeval_loader_mapping")

