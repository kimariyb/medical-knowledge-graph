"""Regression tests for imbalance handling and early stopping."""

from __future__ import annotations

import unittest

import train
from utils import dataloader


class ClassWeightTests(unittest.TestCase):
    def test_rare_labels_receive_larger_training_weights(self) -> None:
        """Weights are calculated from only the labels in the training split."""

        self.assertTrue(hasattr(dataloader, "calculate_class_weights"))
        weights = dataloader.calculate_class_weights(
            [(["a", "b", "c", "d", "e"], [0, 0, 0, 0, 1])],
            {"O": 0, "B-DISEASE": 1},
            power=0.5,
        )
        self.assertGreater(weights[1].item(), weights[0].item())


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_meaningful_f1_improvement(self) -> None:
        """Small F1 fluctuations below min_delta count as no improvement."""

        self.assertTrue(hasattr(train, "EarlyStopping"))
        early_stopping = train.EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(early_stopping.step(0.50))
        self.assertFalse(early_stopping.step(0.505))
        self.assertTrue(early_stopping.step(0.509))


if __name__ == "__main__":
    unittest.main()
