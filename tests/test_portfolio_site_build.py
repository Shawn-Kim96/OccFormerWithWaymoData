import tempfile
import unittest
from pathlib import Path

from tools.build_portfolio_site import build_site
from tools.validate_portfolio_site import validate_site


class PortfolioSiteBuildTest(unittest.TestCase):
    def test_build_and_validate_site(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "site"
            manifest = build_site(output_dir)
            self.assertTrue((output_dir / "index.html").exists())
            self.assertEqual(manifest["project"]["best_experiment"]["experiment"], "baseline_fast")
            report = validate_site(output_dir)
            self.assertTrue(report["technical"]["pass"])
            self.assertGreaterEqual(report["composite_score"], 85)
            self.assertTrue(report["threshold_pass"])


if __name__ == "__main__":
    unittest.main()
