import math
import unittest
import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
from common.generation import _response_mask, response_sequence_logprobs, response_token_logprobs


class Objectives(unittest.TestCase):
    def test_dpo_reference_cancellation(self):
        chosen, rejected = torch.tensor([-1., -5.]), torch.tensor([-3., -2.])
        loss, stats = dpo_loss(chosen, rejected, chosen, rejected, .1)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(float(stats["preference_accuracy"]), 0)

    def test_dpo_gradient_favors_chosen(self):
        chosen = torch.tensor([-1.], requires_grad=True)
        rejected = torch.tensor([-2.], requires_grad=True)
        loss, _ = dpo_loss(chosen, rejected, torch.tensor([-1.]), torch.tensor([-2.]), .3)
        loss.backward()
        self.assertLess(float(chosen.grad), 0)
        self.assertGreater(float(rejected.grad), 0)

    def test_ppo_both_clipping_directions(self):
        ratio = torch.tensor([[1.5, .5, 1.5, .5, 99.]])
        adv = torch.tensor([[1., -1., -1., 1., 100.]])
        mask = torch.tensor([[1., 1., 1., 1., 0.]])
        loss, _, clip = ppo_policy_loss(ratio.log(), torch.zeros_like(ratio), adv, mask, .2)
        self.assertAlmostEqual(float(loss), -((1.2 - .8 - 1.5 + .5) / 4), places=6)
        self.assertEqual(float(clip), 1)

    def test_terminal_gae_and_padding(self):
        mask = torch.tensor([[1., 1., 0.]])
        advantages, returns = compute_gae(torch.tensor([[0., 2., 0.]]), torch.tensor([[.5, 1., 123.]]), mask, 1, 1)
        torch.testing.assert_close(advantages, torch.tensor([[1.5, 1., 0.]]))
        torch.testing.assert_close(returns, torch.tensor([[2., 2., 0.]]))

    def test_kl_shaping_terminal_reward(self):
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
        rewards = shaped_rewards(torch.tensor([2., 3.]), torch.ones_like(mask), torch.zeros_like(mask), mask, .1)
        torch.testing.assert_close(rewards, torch.tensor([[-.1, 1.9, 0.], [2.9, 0., 0.]]))

    def test_grpo_groups_are_independent(self):
        rewards = torch.tensor([1., 3., 101., 103., 5., 5.])
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        adv = group_relative_advantages(rewards, groups)
        torch.testing.assert_close(adv[:2], adv[2:4])
        torch.testing.assert_close(adv[4:], torch.zeros(2))
        self.assertAlmostEqual(float(adv[:2].sum()), 0)

    def test_normalization_and_truncation(self):
        new = torch.zeros((2, 4), requires_grad=True)
        mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
        adv = torch.ones(2)
        canonical, _ = grpo_policy_loss(new, new.detach(), adv, mask, new.detach(), .2, 0, "grpo", 4)
        constant, _ = grpo_policy_loss(new, new.detach(), adv, mask, new.detach(), .2, 0, "dr_grpo", 4)
        self.assertEqual(float(canonical), -1)
        self.assertEqual(float(constant), -.75)
        grad = torch.autograd.grad(canonical, new)[0]
        self.assertEqual(float(grad[0, 0]), -.25)
        self.assertEqual(float(grad[1, 0]), -.125)
        masked = mask_truncated_sequences(mask, [True, False])
        self.assertEqual(float(masked[0].sum()), 0)

    def test_eos_included_padding_excluded(self):
        ids = torch.tensor([[3, 9, 9, 9], [3, 4, 5, 6]])
        torch.testing.assert_close(_response_mask(ids, 9), torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]]))

    def test_response_logprob_alignment(self):
        from types import SimpleNamespace
        class Model:
            def __call__(self, input_ids, **kwargs):
                batch, width = input_ids.shape
                logits = torch.zeros(batch, width, 8)
                for t in range(width-1):
                    logits[:, t, input_ids[0, t+1]] = 4
                return SimpleNamespace(logits=logits)
        batch = {"input_ids": torch.tensor([[0, 2, 3, 4]]), "attention_mask": torch.tensor([[0, 1, 1, 1]]),
                 "response_mask": torch.tensor([[0., 0., 1., 1.]])}
        seq, tokens, mask = response_sequence_logprobs(Model(), batch)
        expected = 4 - math.log(math.exp(4) + 7)
        self.assertAlmostEqual(float(seq[0]), 2 * expected, places=5)
        sampled, _ = response_token_logprobs(Model(), batch["input_ids"], batch["attention_mask"], 2, batch["input_ids"][:, 2:])
        torch.testing.assert_close(sampled.sum(-1), seq)


if __name__ == "__main__":
    unittest.main()
