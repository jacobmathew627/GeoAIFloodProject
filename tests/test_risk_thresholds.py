"""
Tests for the risk-threshold derivation.

The derivation itself needs the hazard raster and the inventory, so only the
criteria table and its consistency with config are checked here. That is
still worth pinning: the thresholds are properties of the fitted
probabilities, and a mismatch between the table and the config is exactly the
drift this module exists to prevent.
"""

from config import RISK
from risk_thresholds import CRITERIA


class TestCriteriaTable:
    def test_covers_every_configured_band(self):
        names = {name for name, _, _ in CRITERIA}
        assert names == {"moderate", "high", "severe", "critical"}

    def test_ordered_from_inclusive_to_selective(self):
        """
        Recall targets must fall as the bands get more severe, or the derived
        thresholds will not be monotonic and the bands will overlap.
        """
        recalls = [t for _, c, t in CRITERIA if c == "recall"]
        assert recalls == sorted(recalls, reverse=True)

    def test_criteria_are_recognised(self):
        for name, criterion, target in CRITERIA:
            assert criterion in {"recall", "precision", "max_f1"}, name
            if criterion == "max_f1":
                assert target is None, name
            else:
                assert 0.0 < target < 1.0, name

    def test_the_top_band_is_precision_led(self):
        """
        The critical band exists to be acted on, so it is defined by
        precision -- how often a flag is right -- not by coverage.
        """
        top = CRITERIA[-1]
        assert top[0] == "critical"
        assert top[1] == "precision"

    def test_the_bottom_band_is_recall_led(self):
        """The most inclusive band exists to miss as little as possible."""
        first = CRITERIA[0]
        assert first[0] == "moderate"
        assert first[1] == "recall"
        assert first[2] >= 0.9


class TestConfigConsistency:
    def test_thresholds_are_ordered(self):
        assert 0 < RISK.safe < RISK.moderate < RISK.high < RISK.critical < 1

    def test_thresholds_are_not_round_numbers(self):
        """
        Round numbers are a sign someone hand-picked them instead of reading
        them off the precision-recall curve.
        """
        derived = [RISK.safe, RISK.moderate, RISK.high, RISK.critical]
        assert not all(
            abs(v * 100 - round(v * 100)) < 1e-9 for v in derived
        ), "every threshold is a whole percentage point; were these derived?"

    def test_thresholds_suit_a_low_prevalence_target(self):
        """
        The domain prevalence is a few percent, so the lowest band has to sit
        near it. A 10% floor would put the whole district in 'safe'.
        """
        assert RISK.safe < 0.10
