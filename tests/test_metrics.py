import unittest

from lyric_detector.metrics import word_error_rate


class WordErrorRateTests(unittest.TestCase):
    def test_perfect_match_is_zero(self):
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)

    def test_single_substitution(self):
        self.assertAlmostEqual(word_error_rate("a b c d", "a b x d"), 0.25)

    def test_insertion(self):
        self.assertAlmostEqual(word_error_rate("a b", "a b c"), 0.5)

    def test_deletion(self):
        self.assertAlmostEqual(word_error_rate("a b c", "a c"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "extra"), 1.0)

    def test_case_matters_by_default(self):
        self.assertGreater(word_error_rate("Hello", "hello"), 0.0)


if __name__ == "__main__":
    unittest.main()
