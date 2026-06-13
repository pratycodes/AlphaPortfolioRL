import unittest

import torch

from SparseNetwork4DRL import SparseLinear, build_mlp
from config.settings import config
from data.universe import feature_dim, ipm_feature_dim
from models.actor_critic import Actor, Critic


class SparseNetworkTest(unittest.TestCase):
    def test_sparse_linear_keeps_fixed_mask_shape(self):
        layer = SparseLinear(10, 4, density=0.3, seed=123)

        self.assertEqual(layer.mask.shape, (4, 10))
        self.assertTrue(torch.all(layer.mask.sum(dim=1) > 0))
        self.assertTrue(torch.all(layer.mask.sum(dim=0) > 0))

    def test_erdos_renyi_topology_allocates_layer_specific_densities(self):
        network = build_mlp(8, [32, 16], 4, use_sparse=True, density=0.5, seed=123, topology="erdos_renyi")
        sparse_layers = [layer for layer in network if isinstance(layer, SparseLinear)]
        densities = [round(layer.density, 4) for layer in sparse_layers]

        self.assertGreater(len(set(densities)), 1)
        self.assertLessEqual(max(densities), 1.0)

    def test_erdos_renyi_aliases_resolve_to_same_density_allocation(self):
        er_network = build_mlp(8, [32, 16], 4, use_sparse=True, density=0.5, seed=123, topology="ER")
        canonical_network = build_mlp(8, [32, 16], 4, use_sparse=True, density=0.5, seed=123, topology="erdos_renyi")

        er_densities = [layer.density for layer in er_network if isinstance(layer, SparseLinear)]
        canonical_densities = [layer.density for layer in canonical_network if isinstance(layer, SparseLinear)]

        self.assertEqual(er_densities, canonical_densities)

    def test_actor_and_critic_forward_with_sparse_heads(self):
        num_assets = len(config.ASSETS)
        action_dim = num_assets + 1
        window_size = config.WINDOW_SIZE
        batch_size = 5

        actor = Actor(num_assets, window_size, action_dim)
        critic = Critic(num_assets, window_size, action_dim)
        model_feature_dim = feature_dim(config)
        state = torch.ones(batch_size, window_size, model_feature_dim)
        prev_weights = torch.zeros(batch_size, action_dim)
        prev_weights[:, 0] = 1.0
        ipm_pred = torch.ones(batch_size, ipm_feature_dim(config))

        action = actor(state, prev_weights, ipm_pred)
        q_value = critic(state, prev_weights, ipm_pred, action)

        self.assertEqual(action.shape, (batch_size, action_dim))
        self.assertEqual(q_value.shape, (batch_size, 1))
        torch.testing.assert_close(action.sum(dim=1), torch.ones(batch_size), atol=1e-6, rtol=1e-6)
        if config.MAX_CASH_WEIGHT is not None:
            self.assertLessEqual(float(action[:, 0].max()), config.MAX_CASH_WEIGHT + 1e-6)
        if config.MAX_WEIGHT is not None:
            self.assertLessEqual(float(action[:, 1:].max()), config.MAX_WEIGHT + 1e-6)
        self.assertFalse(config.USE_SPARSE_NETWORK)


if __name__ == "__main__":
    unittest.main()
