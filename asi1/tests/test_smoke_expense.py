import unittest

from asi1.smoke_expense import MAX_EXAMPLES, MAX_STEPS, MAX_TRAIN_SECONDS, PROBE, TRAIN, example_count


class SmokeExpenseTests(unittest.TestCase):
    def test_dataset_stays_within_ten_examples_and_holds_out_the_probe(self):
        self.assertLessEqual(example_count(), MAX_EXAMPLES)
        self.assertEqual(len(TRAIN), 6)
        self.assertEqual(MAX_STEPS, 6)
        self.assertEqual(MAX_TRAIN_SECONDS, 120)
        self.assertNotIn(PROBE["input"], {case["input"] for case in TRAIN})
        self.assertEqual(PROBE["output"]["category"], "travel")


if __name__ == "__main__":
    unittest.main()
