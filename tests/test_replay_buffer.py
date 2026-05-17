import unittest

import numpy as np

from agent.replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_sample_includes_done_flags(self):
        buffer = ReplayBuffer(capacity=4, batch_size=2, device="cpu")
        state = np.zeros((3, 6), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0])

        buffer.add(state, weights, weights, 0.0, state, weights, weights, done=False)
        buffer.add(state, weights, weights, 1.0, state, weights, weights, done=True)

        *_, done = buffer.sample()
        self.assertEqual(done.dtype, np.float32)
        self.assertEqual(done.shape, (2,))
        self.assertEqual(set(done.tolist()), {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
