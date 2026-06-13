from collections import deque

import numpy as np
import torch

from data.universe import feature_dim


INITIAL_PORTFOLIO_VALUE = 10000.0


class ShadowARBBuffer:
    """
    Portfolio-specific Adaptive Replay Buffer run in shadow mode.

    It receives the same transitions as the uniform replay buffer and computes
    soft importance scores without affecting early training. In this project
    there is no true offline-to-online phase, so ARB activates only after the
    live policy has stabilized on a consistent probe set.

    Inspired by the local ARB.pdf reference, adapted from offline-to-online RL
    into a shadow observer for online portfolio training.
    """

    def __init__(self, capacity, config):
        self.capacity = capacity
        self.records = deque(maxlen=capacity)
        self.counter = 0
        self.initial_portfolio_value = float(getattr(config, "INITIAL_CAPITAL", INITIAL_PORTFOLIO_VALUE))

        self.adaptive_activation = bool(getattr(config, "USE_ADAPTIVE_ARB_ACTIVATION", True))
        self.min_episode = int(getattr(config, "ARB_MIN_EPISODE", 20))
        self.stability_patience = int(getattr(config, "ARB_STABILITY_PATIENCE", 3))
        self.policy_drift_threshold = float(getattr(config, "ARB_POLICY_DRIFT_THRESHOLD", 0.075))
        self.instability_decay = float(getattr(config, "ARB_INSTABILITY_DECAY", 0.50))
        self.stability_recovery = float(getattr(config, "ARB_STABILITY_RECOVERY", 0.20))
        self.min_portfolio_value_ratio = float(getattr(config, "ARB_MIN_PORTFOLIO_VALUE_RATIO", 0.90))
        self.max_activation_turnover = float(getattr(config, "ARB_MAX_ACTIVATION_TURNOVER", 0.18))
        self.min_validation_score = float(getattr(config, "ARB_MIN_VALIDATION_SCORE", -0.25))
        self.probe_size = int(getattr(config, "ARB_PROBE_SIZE", 256))
        self.ramp_episodes = max(int(getattr(config, "ARB_RAMP_EPISODES", 30)), 1)
        self.cache_refresh_interval = max(int(getattr(config, "ARB_CACHE_REFRESH_INTERVAL", 256)), 1)
        self.start_episode = int(getattr(config, "ARB_START_EPISODE", 50))
        self.full_episode = int(getattr(config, "ARB_FULL_EPISODE", 150))
        self.max_mix = float(getattr(config, "ARB_MAX_MIX", 0.80))
        self.temperature = max(float(getattr(config, "ARB_TEMPERATURE", 0.25)), 1e-8)
        self.min_probability = max(float(getattr(config, "ARB_MIN_PROBABILITY", 1e-4)), 0.0)
        self.recency_tau = max(float(getattr(config, "ARB_RECENCY_TAU", 5000.0)), 1.0)

        self.reward_weight = float(getattr(config, "ARB_REWARD_WEIGHT", 0.35))
        self.uncertainty_weight = float(getattr(config, "ARB_UNCERTAINTY_WEIGHT", 0.25))
        self.on_policy_weight = float(getattr(config, "ARB_ON_POLICY_WEIGHT", 0.25))
        self.recency_weight = float(getattr(config, "ARB_RECENCY_WEIGHT", 0.15))

        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.reward_count = 0
        self.activation_episode = None
        self.stable_count = 0
        self.stability_multiplier = 0.0
        self.last_health_ok = True
        self.last_health_reason = "not_observed"
        self.last_health_score = 1.0
        self.last_validation_ok = True
        self.last_validation_score = None
        self.last_policy_drift = None
        self.last_probe_actions = None
        self.probe_records = []
        self._probability_cache = None
        self._cache_counter = None
        self._cache_first_inserted_at = None
        self._probability_cache_dirty = True

    def add(self, transition):
        reward = float(transition[3])
        self._update_reward_stats(reward)
        self.records.append(
            {
                "transition": transition,
                "reward": reward,
                "inserted_at": self.counter,
                "components": self._components(transition, reward),
            }
        )
        self.counter += 1

    def observe_training_health(self, episode_reward, final_value, average_turnover, initial_value=None):
        if initial_value is None:
            initial_value = self.initial_portfolio_value
        value_ratio = float(final_value) / max(float(initial_value), 1e-8)
        turnover = float(average_turnover)
        value_ok = value_ratio >= self.min_portfolio_value_ratio
        turnover_ok = turnover <= self.max_activation_turnover

        self.last_health_ok = bool(value_ok and turnover_ok)
        self.last_health_score = value_ratio
        if not value_ok:
            self.last_health_reason = "portfolio_value"
        elif not turnover_ok:
            self.last_health_reason = "turnover"
        else:
            self.last_health_reason = "ok"

        if not self.last_health_ok and self.activation_episode is not None:
            self.stability_multiplier *= self.instability_decay
        return self.health_diagnostics()

    def observe_validation_health(self, validation_score):
        if validation_score is None:
            return self.health_diagnostics()

        self.last_validation_score = float(validation_score)
        self.last_validation_ok = self.last_validation_score >= self.min_validation_score
        if not self.last_validation_ok and self.activation_episode is not None:
            self.stability_multiplier *= self.instability_decay
        return self.health_diagnostics()

    def mix_ratio(self, episode):
        if self.adaptive_activation:
            if self.activation_episode is None or episode is None:
                return 0.0
            progress = (episode - self.activation_episode) / self.ramp_episodes
            scheduled_mix = float(np.clip(progress, 0.0, 1.0) * self.max_mix)
            return scheduled_mix * self.stability_multiplier

        if episode is None or episode < self.start_episode:
            return 0.0
        if self.full_episode <= self.start_episode:
            return self.max_mix

        progress = (episode - self.start_episode) / (self.full_episode - self.start_episode)
        return float(np.clip(progress, 0.0, 1.0) * self.max_mix)

    def observe_policy(self, agent, ipm_model, device, config, episode):
        if not self.adaptive_activation or len(self.records) == 0:
            return self.diagnostics()

        self._refresh_probe_records()
        if not self.probe_records:
            return self.diagnostics()

        actions = self._policy_actions(agent, ipm_model, device, config, self.probe_records)
        if self.last_probe_actions is None:
            self.last_probe_actions = actions
            self.last_policy_drift = None
            return self.diagnostics()

        drift = float(np.mean(np.sum(np.abs(actions - self.last_probe_actions), axis=1)))
        self.last_policy_drift = drift
        self.last_probe_actions = actions
        self._update_probe_on_policy_scores(actions)
        self._invalidate_probability_cache()
        self.observe_stability(drift, episode)
        return self.diagnostics()

    def observe_stability(self, policy_drift, episode):
        if episode < self.min_episode:
            self.stable_count = 0
            return

        if policy_drift <= self.policy_drift_threshold and self._activation_health_ok():
            self.stable_count += 1
            if self.activation_episode is not None:
                self.stability_multiplier = min(1.0, self.stability_multiplier + self.stability_recovery)
        else:
            self.stable_count = 0
            if self.activation_episode is not None:
                self.stability_multiplier *= self.instability_decay

        if self.activation_episode is None and self.stable_count >= self.stability_patience:
            self.activation_episode = episode
            self.stability_multiplier = 1.0

    def diagnostics(self):
        return {
            "activation_episode": self.activation_episode,
            "stable_count": self.stable_count,
            "stability_multiplier": self.stability_multiplier,
            "health_ok": self._activation_health_ok(),
            "health_reason": self._health_reason(),
            "health_score": self.last_health_score,
            "validation_ok": self.last_validation_ok,
            "validation_score": self.last_validation_score,
            "policy_drift": self.last_policy_drift,
            "policy_drift_threshold": self.policy_drift_threshold,
        }

    def health_diagnostics(self):
        return {
            "health_ok": self._activation_health_ok(),
            "health_reason": self._health_reason(),
            "health_score": self.last_health_score,
            "validation_ok": self.last_validation_ok,
            "validation_score": self.last_validation_score,
            "min_portfolio_value_ratio": self.min_portfolio_value_ratio,
            "initial_portfolio_value": self.initial_portfolio_value,
            "max_activation_turnover": self.max_activation_turnover,
            "min_validation_score": self.min_validation_score,
        }

    def _activation_health_ok(self):
        return bool(self.last_health_ok and self.last_validation_ok)

    def _health_reason(self):
        if not self.last_health_ok:
            return self.last_health_reason
        if not self.last_validation_ok:
            return "validation"
        return "ok"

    def sample(self, batch_size):
        if not self.records:
            return []

        probabilities = self.probabilities()
        indices = np.random.choice(
            len(self.records),
            size=batch_size,
            replace=len(self.records) < batch_size,
            p=probabilities,
        )
        return [self.records[index]["transition"] for index in indices]

    def probabilities(self):
        cached = self._cached_probabilities()
        if cached is not None:
            return cached

        scores = np.asarray([self._score(record) for record in self.records], dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if np.all(scores <= 0):
            probabilities = np.full(len(scores), 1.0 / len(scores))
            self._store_probability_cache(probabilities)
            return probabilities

        logits = scores / self.temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities += self.min_probability
        probabilities /= probabilities.sum()
        self._store_probability_cache(probabilities)
        return probabilities

    def score_snapshot(self, limit=10):
        if not self.records:
            return []

        probabilities = self.probabilities()
        rows = []
        for index, record in enumerate(self.records):
            rows.append(
                {
                    "index": index,
                    "probability": float(probabilities[index]),
                    "score": float(self._score(record)),
                    "reward": float(record["reward"]),
                    "age": int(self.counter - record["inserted_at"]),
                }
            )

        return sorted(rows, key=lambda row: row["probability"], reverse=True)[:limit]

    def _components(self, transition, reward):
        _, prev_weights, action, _, _, _, greedy_action, _ = transition
        action = np.asarray(action, dtype=float)
        prev_weights = np.asarray(prev_weights, dtype=float)
        greedy_action = np.asarray(greedy_action, dtype=float)

        turnover_proxy = float(np.sum(np.abs(action - prev_weights)))
        return {
            "reward": self._reward_score(reward),
            "uncertainty": self._uncertainty_score(reward, action, greedy_action),
            "on_policy": self._on_policy_score(action, greedy_action),
            "turnover_proxy": turnover_proxy,
        }

    def _score(self, record):
        age = max(self.counter - record["inserted_at"], 0)
        recency_score = np.exp(-age / self.recency_tau)
        components = record["components"]
        score = (
            self.reward_weight * components["reward"]
            + self.uncertainty_weight * components["uncertainty"]
            + self.on_policy_weight * components["on_policy"]
            + self.recency_weight * recency_score
        )
        return max(score / (1.0 + 0.05 * components["turnover_proxy"]), 1e-8)

    def _cached_probabilities(self):
        if self._probability_cache is None or self._probability_cache_dirty:
            return None
        if not self.records:
            return None
        if self._cache_first_inserted_at != self.records[0]["inserted_at"]:
            return None

        record_count = len(self.records)
        cached_count = len(self._probability_cache)
        if cached_count == record_count:
            return self._probability_cache
        if cached_count > record_count:
            return None
        if record_count - cached_count >= self.cache_refresh_interval:
            return None

        new_count = record_count - cached_count
        probabilities = np.concatenate(
            [
                self._probability_cache * (cached_count / record_count),
                np.full(new_count, 1.0 / record_count),
            ]
        )
        probabilities /= probabilities.sum()
        self._store_probability_cache(probabilities)
        return probabilities

    def _store_probability_cache(self, probabilities):
        self._probability_cache = probabilities
        self._cache_counter = self.counter
        self._cache_first_inserted_at = self.records[0]["inserted_at"] if self.records else None
        self._probability_cache_dirty = False

    def _invalidate_probability_cache(self):
        self._probability_cache_dirty = True

    def _refresh_probe_records(self):
        target_size = min(self.probe_size, len(self.records))
        if target_size <= 0:
            self.probe_records = []
            self.last_probe_actions = None
            return

        if self.probe_records and len(self.probe_records) == target_size:
            return

        previous_signature = self._probe_signature(self.probe_records)
        records = list(self.records)
        candidates = list(self.probe_records[:target_size])
        seen = {record["inserted_at"] for record in candidates}

        if len(records) <= target_size:
            candidate_pool = records
        else:
            tail_span = min(len(records), max(target_size * 4, target_size))
            start_index = len(records) - tail_span
            indices = np.linspace(start_index, len(records) - 1, num=target_size, dtype=int)
            candidate_pool = [records[index] for index in indices]

        for record in candidate_pool:
            if len(candidates) >= target_size:
                break
            if record["inserted_at"] not in seen:
                candidates.append(record)
                seen.add(record["inserted_at"])

        if len(candidates) < target_size:
            for record in records:
                if len(candidates) >= target_size:
                    break
                if record["inserted_at"] not in seen:
                    candidates.append(record)
                    seen.add(record["inserted_at"])

        if self._probe_signature(candidates) != previous_signature:
            self.last_probe_actions = None
            self.last_policy_drift = None
            self.stable_count = 0
            self.probe_records = candidates
            return

        self.probe_records = candidates

    def _probe_signature(self, records):
        return tuple(record["inserted_at"] for record in records)

    def _policy_actions(self, agent, ipm_model, device, config, records):
        states = np.asarray([record["transition"][0] for record in records], dtype=np.float32)
        prev_weights = np.asarray([record["transition"][1] for record in records], dtype=np.float32)

        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        prev_weight_tensor = torch.as_tensor(prev_weights, dtype=torch.float32, device=device)

        actor_was_training = agent.actor.training
        ipm_was_training = ipm_model.training
        agent.actor.eval()
        ipm_model.eval()

        with torch.no_grad():
            if getattr(config, "USE_IPM", False):
                ipm_pred = ipm_model(state_tensor)
            else:
                ipm_pred = torch.zeros(state_tensor.size(0), feature_dim(config), device=device)
            actions = agent.actor(state_tensor, prev_weight_tensor, ipm_pred).detach().cpu().numpy()

        if actor_was_training:
            agent.actor.train()
        if ipm_was_training:
            ipm_model.train()

        return actions

    def _update_probe_on_policy_scores(self, current_actions):
        for record, current_action in zip(self.probe_records, current_actions):
            stored_action = np.asarray(record["transition"][2], dtype=float)
            record["components"]["on_policy"] = self._on_policy_score(current_action, stored_action)

    def _reward_score(self, reward):
        std = self._reward_std()
        z_score = (reward - self.reward_mean) / std
        return float(1.0 / (1.0 + np.exp(-z_score)))

    def _uncertainty_score(self, reward, action, greedy_action):
        std = self._reward_std()
        reward_surprise = min(abs(reward - self.reward_mean) / (3.0 * std), 1.0)
        policy_disagreement = min(float(np.sum(np.abs(action - greedy_action))) / 2.0, 1.0)
        return 0.5 * reward_surprise + 0.5 * policy_disagreement

    def _on_policy_score(self, action, greedy_action):
        distance = float(np.sum(np.abs(action - greedy_action)))
        return float(np.exp(-distance))

    def _update_reward_stats(self, reward):
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

    def _reward_std(self):
        if self.reward_count < 2:
            return 1.0
        return max(float(np.sqrt(self.reward_m2 / (self.reward_count - 1))), 1e-6)

    def __len__(self):
        return len(self.records)
