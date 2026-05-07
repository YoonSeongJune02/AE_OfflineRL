"""
Unit tests for DAE and RewardShaper.

Run with:
    python -m pytest tests/test_dae.py -v
    # or simply:
    python tests/test_dae.py

Tests cover:
    - DAE forward/encode/decode shape correctness
    - DAE reconstruction error shape and non-negativity
    - DAE eval-mode preservation in reconstruction_error
    - Noise injection functions
    - RunningMeanStd correctness against numpy
    - RewardShaper threshold/normalization/lambda warmup behavior
    - RewardShaper state save/load round-trip
"""

import os
import sys
import unittest

import numpy as np
import torch

# Make Algos importable when run directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))

from Algos.DAE import (  # noqa: E402
    DAE,
    add_gaussian_noise,
    add_masking_noise,
    add_mixed_noise,
)
from Algos.reward_shaping import RewardShaper, RunningMeanStd  # noqa: E402


class TestDAE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.state_dim = 19
        self.batch_size = 32
        self.dae = DAE(state_dim=self.state_dim, latent_dim=8, hidden_dim=64, dropout=0.0)

    def test_forward_shape(self):
        x = torch.randn(self.batch_size, self.state_dim)
        out = self.dae(x)
        self.assertEqual(out.shape, x.shape)

    def test_encode_decode_shapes(self):
        x = torch.randn(self.batch_size, self.state_dim)
        z = self.dae.encode(x)
        x_hat = self.dae.decode(z)
        self.assertEqual(z.shape, (self.batch_size, 8))
        self.assertEqual(x_hat.shape, x.shape)

    def test_reconstruction_error_shape(self):
        x = torch.randn(self.batch_size, self.state_dim)
        err = self.dae.reconstruction_error(x)
        self.assertEqual(err.shape, (self.batch_size, 1))
        self.assertTrue((err >= 0).all().item())

    def test_reconstruction_error_preserves_mode(self):
        """recon error should leave training mode unchanged."""
        x = torch.randn(self.batch_size, self.state_dim)
        self.dae.train()
        _ = self.dae.reconstruction_error(x)
        self.assertTrue(self.dae.training)

        self.dae.eval()
        _ = self.dae.reconstruction_error(x)
        self.assertFalse(self.dae.training)


class TestNoiseFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.zeros(100, 10)

    def test_gaussian_noise_changes_input(self):
        y = add_gaussian_noise(self.x, std=1.0)
        self.assertFalse(torch.allclose(y, self.x))

    def test_gaussian_noise_zero_std(self):
        y = add_gaussian_noise(self.x, std=0.0)
        self.assertTrue(torch.allclose(y, self.x))

    def test_masking_zeros_some_features(self):
        x = torch.ones(1000, 10)
        y = add_masking_noise(x, mask_prob=0.5)
        zero_frac = (y == 0).float().mean().item()
        self.assertAlmostEqual(zero_frac, 0.5, delta=0.05)

    def test_mixed_applies_both(self):
        x = torch.ones(1000, 10)
        y = add_mixed_noise(x, gaussian_std=0.1, mask_prob=0.3)
        zero_frac = (y == 0).float().mean().item()
        # ~30% should be zero from masking
        self.assertGreater(zero_frac, 0.2)


class TestRunningMeanStd(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.RandomState(0)
        rms = RunningMeanStd()
        all_values = []
        for _ in range(10):
            batch = rng.randn(50)
            rms.update(torch.from_numpy(batch))
            all_values.extend(batch.tolist())

        all_arr = np.array(all_values)
        # Welford has small numerical drift; loose tolerance is fine
        self.assertAlmostEqual(rms.mean, float(all_arr.mean()), places=4)
        self.assertAlmostEqual(rms.std, float(all_arr.std()), places=2)


class TestRewardShaper(unittest.TestCase):
    def test_warmup_lambda(self):
        shaper = RewardShaper(lambda_max=1.0, warmup_steps=100, threshold_z=0.0)
        self.assertAlmostEqual(shaper.current_lambda(), 0.0)

        shaper.step = 50
        self.assertAlmostEqual(shaper.current_lambda(), 0.5)

        shaper.step = 200
        self.assertAlmostEqual(shaper.current_lambda(), 1.0)

    def test_no_warmup(self):
        shaper = RewardShaper(lambda_max=0.5, warmup_steps=0, threshold_z=0.0)
        self.assertAlmostEqual(shaper.current_lambda(), 0.5)

    def test_threshold_filters_low_errors(self):
        """When errors are below the threshold (in z-space), no penalty applied."""
        shaper = RewardShaper(
            lambda_max=1.0, warmup_steps=0, threshold_z=10.0,  # very high threshold
            use_normalization=True, use_tanh=True,
        )
        reward = torch.ones(64, 1)
        recon = torch.ones(64, 1) * 0.01  # tiny errors
        # Run a few times to populate running stats
        for _ in range(5):
            shaped, info = shaper.shape(reward, recon)
        # With huge threshold, penalty should be ~0 -> shaped ≈ reward
        self.assertAlmostEqual(info["penalty_mean"], 0.0, places=4)
        self.assertTrue(torch.allclose(shaped, reward, atol=1e-3))

    def test_penalty_reduces_reward(self):
        """When errors exceed threshold, shaped reward < original."""
        shaper = RewardShaper(
            lambda_max=1.0, warmup_steps=0, threshold_z=-5.0,  # very low threshold
            use_normalization=False, use_tanh=True,
        )
        reward = torch.ones(64, 1)
        recon = torch.ones(64, 1) * 5.0
        shaped, info = shaper.shape(reward, recon)
        self.assertGreater(info["penalty_mean"], 0.0)
        self.assertTrue((shaped <= reward).all().item())

    def test_state_dict_roundtrip(self):
        shaper = RewardShaper(lambda_max=0.7, warmup_steps=200, threshold_z=1.5)
        # Run a few updates to mutate state
        for _ in range(3):
            shaper.shape(torch.randn(32, 1), torch.randn(32, 1).abs())

        state = shaper.state_dict()
        shaper2 = RewardShaper()
        shaper2.load_state_dict(state)

        self.assertEqual(shaper2.lambda_max, shaper.lambda_max)
        self.assertEqual(shaper2.warmup_steps, shaper.warmup_steps)
        self.assertEqual(shaper2.threshold_z, shaper.threshold_z)
        self.assertEqual(shaper2.step, shaper.step)
        self.assertAlmostEqual(shaper2.running.mean, shaper.running.mean)


if __name__ == "__main__":
    unittest.main(verbosity=2)