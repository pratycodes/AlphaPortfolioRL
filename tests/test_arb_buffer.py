import unittest
from types import SimpleNamespace

import numpy as np

from agent.replay_buffer import ReplayBuffer


def make_arb_config():
    return SimpleNamespace(
        USE_ARB=True,
        ARB_START_EPISODE=10,
        ARB_FULL_EPISODE=20,
        ARB_MAX_MIX=0.8,
        ARB_TEMPERATURE=0.25,
        ARB_MIN_PROBABILITY=1e-4,
        ARB_RECENCY_TAU=100.0,
        ARB_REWARD_WEIGHT=0.35,
        ARB_UNCERTAINTY_WEIGHT=0.25,
        ARB_ON_POLICY_WEIGHT=0.25,
        ARB_RECENCY_WEIGHT=0.15,
    )


class ShadowARBReplayTest(unittest.TestCase):
    def test_arb_stays_shadow_until_warmup_finishes(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config())

        self.assertEqual(buffer.current_arb_mix(episode=5), 0.0)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=15), 0.4)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=20), 0.8)

    def test_sample_keeps_standard_batch_contract_when_arb_is_active(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config())
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])

        for index in range(8):
            action = np.array([0.05, 0.45, 0.50])
            buffer.add(state, weights, action, float(index), state, action, action, done=False)

        state_batch, *_, done_batch = buffer.sample(episode=20)

        self.assertEqual(state_batch.shape, (4, 3, 6))
        self.assertEqual(done_batch.shape, (4,))
        self.assertEqual(len(buffer.arb_snapshot(limit=3)), 3)


if __name__ == "__main__":
    unittest.main()
