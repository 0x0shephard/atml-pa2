import unittest
from common.data import filter_preference_prompts


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(int(messages[0]["content"])))


class PromptFilter(unittest.TestCase):
    def test_preserves_prompt_and_reserves_content_and_eos(self):
        rows = [{"chosen": [{"role": "user", "content": str(n)},
                            {"role": "assistant", "content": "yes"}],
                 "rejected": [{"role": "user", "content": str(n)},
                              {"role": "assistant", "content": "no"}]}
                for n in (4, 6, 7, 8, 1072)]
        kept, coverage = filter_preference_prompts(Tokenizer(), rows, 8)
        self.assertEqual(kept, rows[:2])
        self.assertEqual(coverage["retained"], 2)
        self.assertEqual([r["source_index"] for r in coverage["excluded"]], [2, 3, 4])
        self.assertEqual(coverage["total"], 5)


if __name__ == "__main__":
    unittest.main()
