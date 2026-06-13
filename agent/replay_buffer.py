import numpy as np
import random
from collections import deque

from ARB import ShadowARBBuffer
from config.settings import config as default_config


class ReplayBuffer:
    def __init__(self, capacity, batch_size, device, config=None):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
        self.config = config or default_config
        self.current_episode = 0
        self.use_arb = bool(getattr(self.config, "USE_ARB", False))
        self.use_prioritized = bool(getattr(self.config, "USE_PRIORITIZED_REPLAY", False)) and not self.use_arb
        self.priority_alpha = float(getattr(self.config, "PRIORITY_ALPHA", 0.6))
        self.priority_beta_start = float(getattr(self.config, "PRIORITY_BETA_START", 0.4))
        self.priority_beta_frames = max(int(getattr(self.config, "PRIORITY_BETA_FRAMES", 100000)), 1)
        self.priority_epsilon = float(getattr(self.config, "PRIORITY_EPSILON", 1e-6))
        self.sample_count = 0
        self.arb = ShadowARBBuffer(capacity, self.config) if self.use_arb else None
        self.last_sample_mix = 0.0

    def add(self, state, prev_weights, action, reward, next_state, next_prev_weights, greedy_action, done=False):
        transition = (state, prev_weights, action, reward, next_state, next_prev_weights, greedy_action, done)
        self.buffer.append(transition)
        if self.use_prioritized:
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(float(max_priority))
        if self.arb is not None:
            self.arb.add(transition)

    def set_training_progress(self, episode):
        self.current_episode = episode

    def current_arb_mix(self, episode=None):
        if self.arb is None:
            return 0.0
        return self.arb.mix_ratio(self.current_episode if episode is None else episode)

    def observe_policy(self, agent, ipm_model, episode=None):
        if self.arb is None:
            return {}
        effective_episode = self.current_episode if episode is None else episode
        return self.arb.observe_policy(agent, ipm_model, self.device, self.config, effective_episode)

    def observe_training_health(self, episode_reward, final_value, average_turnover):
        if self.arb is None:
            return {}
        return self.arb.observe_training_health(episode_reward, final_value, average_turnover)

    def observe_validation_health(self, validation_score):
        if self.arb is None:
            return {}
        return self.arb.observe_validation_health(validation_score)

    def sample(self, episode=None):
        if self.use_prioritized:
            return self._sample_prioritized()

        arb_mix = self.current_arb_mix(episode)
        self.last_sample_mix = arb_mix

        if self.arb is None or arb_mix <= 0.0:
            batch = random.sample(self.buffer, self.batch_size)
            return self._format_batch(batch)

        arb_size = int(round(self.batch_size * arb_mix))
        arb_size = min(max(arb_size, 0), self.batch_size)
        uniform_size = self.batch_size - arb_size

        batch = []
        if uniform_size > 0:
            batch.extend(random.sample(self.buffer, uniform_size))
        if arb_size > 0:
            batch.extend(self.arb.sample(arb_size))

        random.shuffle(batch)
        return self._format_batch(batch)

    def update_priorities(self, indices, td_errors):
        if not self.use_prioritized:
            return

        for index, error in zip(indices, td_errors):
            if 0 <= int(index) < len(self.priorities):
                self.priorities[int(index)] = float(abs(error)) + self.priority_epsilon

    def arb_snapshot(self, limit=10):
        if self.arb is None:
            return []
        return self.arb.score_snapshot(limit=limit)

    def arb_diagnostics(self):
        if self.arb is None:
            return {}
        return self.arb.diagnostics()

    def _sample_prioritized(self):
        priorities = np.asarray(self.priorities, dtype=float)
        if len(priorities) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        scaled = np.power(priorities + self.priority_epsilon, self.priority_alpha)
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(
            len(self.buffer),
            size=self.batch_size,
            replace=len(self.buffer) < self.batch_size,
            p=probabilities,
        )
        self.sample_count += 1
        beta_progress = min(1.0, self.sample_count / self.priority_beta_frames)
        beta = self.priority_beta_start + beta_progress * (1.0 - self.priority_beta_start)
        weights = np.power(len(self.buffer) * probabilities[indices], -beta)
        weights = weights / (weights.max() + 1e-8)
        batch = [self.buffer[int(index)] for index in indices]
        return self._format_batch(batch, indices=indices, importance_weights=weights)

    def _format_batch(self, batch, indices=None, importance_weights=None):
        state, prev_w, action, reward, next_state, next_prev_w, greedy_action, done = zip(*batch)

        formatted = (
            np.array(state),
            np.array(prev_w),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(next_prev_w),
            np.array(greedy_action),
            np.array(done, dtype=np.float32)
        )
        if indices is None:
            return formatted
        return formatted + (
            np.array(indices, dtype=np.int64),
            np.array(importance_weights, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
