import unittest

import torch

from config.settings import config
from data.universe import feature_dim, ipm_feature_dim
from models.ipm import IPM


class IPMTest(unittest.TestCase):
    def test_online_update_changes_trainable_parameters(self):
        device = torch.device("cpu")
        ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(device)
        optimizer = torch.optim.Adam(ipm.parameters(), lr=1e-3)
        state = torch.ones(config.WINDOW_SIZE, feature_dim(config), dtype=torch.float32)
        target = torch.full((ipm_feature_dim(config),), 0.01, dtype=torch.float32)

        before = [param.detach().clone() for param in ipm.parameters()]
        loss = IPM.online_update(ipm, optimizer, state, target, device)
        changed = any(not torch.equal(old, new) for old, new in zip(before, ipm.parameters()))

        self.assertGreaterEqual(loss, 0.0)
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
