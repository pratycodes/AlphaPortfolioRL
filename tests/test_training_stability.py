import unittest
from types import SimpleNamespace

import torch

from agent.ddpg_agent import DDPGAgent
from config.settings import config
from main import _make_evaluation_ipm
from models.ipm import IPM


class TrainingStabilityTest(unittest.TestCase):
    def test_online_evaluation_uses_cloned_ipm(self):
        device = torch.device("cpu")
        ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(device)
        eval_config = SimpleNamespace(USE_IPM=True, USE_ONLINE_IPM=True)

        eval_ipm, cloned = _make_evaluation_ipm(ipm, device, update_ipm_online=True, cfg=eval_config)

        self.assertTrue(cloned)
        self.assertIsNot(eval_ipm, ipm)
        with torch.no_grad():
            next(eval_ipm.parameters()).add_(1.0)
        self.assertFalse(torch.equal(next(eval_ipm.parameters()), next(ipm.parameters())))

    def test_evaluation_ipm_reuses_original_when_online_updates_disabled(self):
        device = torch.device("cpu")
        ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(device)
        eval_config = SimpleNamespace(USE_IPM=True, USE_ONLINE_IPM=True)

        eval_ipm, cloned = _make_evaluation_ipm(ipm, device, update_ipm_online=False, cfg=eval_config)

        self.assertFalse(cloned)
        self.assertIs(eval_ipm, ipm)

    def test_target_mode_context_restores_training_flags(self):
        agent = DDPGAgent(num_assets=len(config.ASSETS), window_size=config.WINDOW_SIZE)
        agent.actor_target.train()
        agent.critic_target.train()

        with agent._temporary_eval(agent.actor_target, agent.critic_target):
            self.assertFalse(agent.actor_target.training)
            self.assertFalse(agent.critic_target.training)

        self.assertTrue(agent.actor_target.training)
        self.assertTrue(agent.critic_target.training)


if __name__ == "__main__":
    unittest.main()
