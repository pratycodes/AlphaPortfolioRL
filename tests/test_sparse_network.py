import unittest

import torch

from SparseNetwork4DRL import SparseLinear
from config.settings import config
from models.actor_critic import Actor, Critic


class SparseNetworkTest(unittest.TestCase):
    def test_sparse_linear_keeps_fixed_mask_shape(self):
        layer = SparseLinear(10, 4, density=0.3, seed=123)

        self.assertEqual(layer.mask.shape, (4, 10))
        self.assertTrue(torch.all(layer.mask.sum(dim=1) > 0))
        self.assertTrue(torch.all(layer.mask.sum(dim=0) > 0))

    def test_actor_and_critic_forward_with_sparse_heads(self):
        num_assets = 2
        action_dim = num_assets + 1
        window_size = 3
        batch_size = 5

        actor = Actor(num_assets, window_size, action_dim)
        critic = Critic(num_assets, window_size, action_dim)
        state = torch.ones(batch_size, window_size, num_assets * 3)
        prev_weights = torch.zeros(batch_size, action_dim)
        prev_weights[:, 0] = 1.0
        ipm_pred = torch.ones(batch_size, num_assets * 3)

        action = actor(state, prev_weights, ipm_pred)
        q_value = critic(state, prev_weights, ipm_pred, action)

        self.assertEqual(action.shape, (batch_size, action_dim))
        self.assertEqual(q_value.shape, (batch_size, 1))
        torch.testing.assert_close(action.sum(dim=1), torch.ones(batch_size), atol=1e-6, rtol=1e-6)
        self.assertTrue(config.USE_SPARSE_NETWORK)


if __name__ == "__main__":
    unittest.main()
