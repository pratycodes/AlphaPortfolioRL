import unittest
from types import SimpleNamespace

import numpy as np

from agent.replay_buffer import ReplayBuffer


def make_arb_config(adaptive=False, **overrides):
    values = dict(
        USE_ARB=True,
        INITIAL_CAPITAL=10000.0,
        USE_ADAPTIVE_ARB_ACTIVATION=adaptive,
        ARB_MIN_EPISODE=5,
        ARB_STABILITY_PATIENCE=2,
        ARB_POLICY_DRIFT_THRESHOLD=0.05,
        ARB_INSTABILITY_DECAY=0.5,
        ARB_STABILITY_RECOVERY=0.2,
        ARB_MIN_PORTFOLIO_VALUE_RATIO=0.9,
        ARB_MAX_ACTIVATION_TURNOVER=0.18,
        ARB_MIN_VALIDATION_SCORE=-0.25,
        ARB_PROBE_SIZE=4,
        ARB_RAMP_EPISODES=10,
        ARB_CACHE_REFRESH_INTERVAL=4,
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
    values.update(overrides)
    return SimpleNamespace(**values)


class ShadowARBReplayTest(unittest.TestCase):
    def test_arb_stays_shadow_until_warmup_finishes(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=False))

        self.assertEqual(buffer.current_arb_mix(episode=5), 0.0)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=15), 0.4)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=20), 0.8)

    def test_sample_keeps_standard_batch_contract_when_arb_is_active(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=False))
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])

        for index in range(8):
            action = np.array([0.05, 0.45, 0.50])
            buffer.add(state, weights, action, float(index), state, action, action, done=False)

        state_batch, *_, done_batch = buffer.sample(episode=20)

        self.assertEqual(state_batch.shape, (4, 3, 6))
        self.assertEqual(done_batch.shape, (4,))
        self.assertEqual(len(buffer.arb_snapshot(limit=3)), 3)

    def test_adaptive_arb_activates_only_after_policy_stabilizes(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))

        buffer.arb.observe_stability(policy_drift=0.10, episode=5)
        self.assertEqual(buffer.current_arb_mix(episode=5), 0.0)

        buffer.arb.observe_stability(policy_drift=0.01, episode=6)
        self.assertEqual(buffer.current_arb_mix(episode=6), 0.0)

        buffer.arb.observe_stability(policy_drift=0.01, episode=7)
        self.assertEqual(buffer.arb.diagnostics()["activation_episode"], 7)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=12), 0.4)

    def test_active_arb_mix_backs_off_when_policy_destabilizes(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))

        buffer.arb.observe_stability(policy_drift=0.01, episode=6)
        buffer.arb.observe_stability(policy_drift=0.01, episode=7)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=12), 0.4)

        buffer.arb.observe_stability(policy_drift=0.20, episode=13)

        self.assertAlmostEqual(buffer.arb.diagnostics()["stability_multiplier"], 0.5)
        self.assertAlmostEqual(buffer.current_arb_mix(episode=13), 0.24)

    def test_adaptive_arb_does_not_activate_on_bad_stable_policy(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))
        buffer.observe_training_health(episode_reward=-100.0, final_value=800.0, average_turnover=0.25)

        buffer.arb.observe_stability(policy_drift=0.00, episode=6)
        buffer.arb.observe_stability(policy_drift=0.00, episode=7)
        buffer.arb.observe_stability(policy_drift=0.00, episode=8)

        self.assertIsNone(buffer.arb.diagnostics()["activation_episode"])
        self.assertEqual(buffer.arb.diagnostics()["stable_count"], 0)
        self.assertFalse(buffer.arb.diagnostics()["health_ok"])
        self.assertEqual(buffer.current_arb_mix(episode=12), 0.0)

    def test_adaptive_arb_health_uses_configured_initial_capital(self):
        buffer = ReplayBuffer(
            capacity=16,
            batch_size=4,
            device="cpu",
            config=make_arb_config(adaptive=True, INITIAL_CAPITAL=500000.0),
        )

        diagnostics = buffer.observe_training_health(
            episode_reward=-10.0,
            final_value=400000.0,
            average_turnover=0.01,
        )

        self.assertFalse(diagnostics["health_ok"])
        self.assertEqual(diagnostics["health_reason"], "portfolio_value")
        self.assertAlmostEqual(diagnostics["health_score"], 0.8)
        self.assertEqual(diagnostics["initial_portfolio_value"], 500000.0)

    def test_adaptive_arb_does_not_activate_when_validation_is_weak(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))
        buffer.observe_training_health(episode_reward=10.0, final_value=12000.0, average_turnover=0.10)
        buffer.observe_validation_health(validation_score=-0.50)

        buffer.arb.observe_stability(policy_drift=0.00, episode=6)
        buffer.arb.observe_stability(policy_drift=0.00, episode=7)
        buffer.arb.observe_stability(policy_drift=0.00, episode=8)

        diagnostics = buffer.arb.diagnostics()
        self.assertIsNone(diagnostics["activation_episode"])
        self.assertEqual(diagnostics["health_reason"], "validation")
        self.assertEqual(buffer.current_arb_mix(episode=12), 0.0)

    def test_cached_arb_probabilities_extend_for_new_records(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=False))
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])
        action = np.array([0.05, 0.45, 0.50])

        for index in range(4):
            buffer.add(state, weights, action, float(index), state, action, action, done=False)

        first = buffer.arb.probabilities()
        buffer.add(state, weights, action, 4.0, state, action, action, done=False)
        second = buffer.arb.probabilities()

        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 5)
        self.assertAlmostEqual(float(second.sum()), 1.0)

    def test_arb_probe_set_stays_fixed_while_measuring_drift(self):
        buffer = ReplayBuffer(capacity=16, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])
        action = np.array([0.05, 0.45, 0.50])

        for index in range(8):
            buffer.add(state, weights, action, float(index), state, action, action, done=False)
        buffer.arb._refresh_probe_records()
        first_signature = buffer.arb._probe_signature(buffer.arb.probe_records)

        for index in range(8, 12):
            buffer.add(state, weights, action, float(index), state, action, action, done=False)
        buffer.arb._refresh_probe_records()
        second_signature = buffer.arb._probe_signature(buffer.arb.probe_records)

        self.assertEqual(first_signature, second_signature)

    def test_probe_set_survives_normal_replay_eviction(self):
        buffer = ReplayBuffer(capacity=32, batch_size=4, device="cpu", config=make_arb_config(adaptive=True))
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])
        action = np.array([0.05, 0.45, 0.50])

        for index in range(32):
            buffer.add(state, weights, action, float(index), state, action, action, done=False)
        buffer.arb._refresh_probe_records()
        first_signature = buffer.arb._probe_signature(buffer.arb.probe_records)

        for index in range(32, 36):
            buffer.add(state, weights, action, float(index), state, action, action, done=False)
        buffer.arb._refresh_probe_records()
        second_signature = buffer.arb._probe_signature(buffer.arb.probe_records)

        self.assertEqual(first_signature, second_signature)


if __name__ == "__main__":
    unittest.main()
