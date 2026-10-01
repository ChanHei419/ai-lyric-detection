import unittest

from lyric_detector.text import clean_lyrics, format_lyrics, merge_repeated_lines


class CleanLyricsTests(unittest.TestCase):
    def test_removes_punctuation_and_lowercases(self):
        self.assertEqual(
            clean_lyrics("Hello, WORLD! It's me."), "hello world it's me"
        )

    def test_collapses_whitespace(self):
        self.assertEqual(clean_lyrics("  too   many    spaces "), "too many spaces")

    def test_replaces_pipe_separators(self):
        self.assertEqual(clean_lyrics("a|b"), "a b")


class MergeRepeatedLinesTests(unittest.TestCase):
    def test_collapses_consecutive_duplicates_only(self):
        self.assertEqual(merge_repeated_lines(["a", "a", "b", "a"]), ["a", "b", "a"])


class FormatLyricsTests(unittest.TestCase):
    def test_wraps_long_lines_without_breaking_words(self):
        formatted = format_lyrics("one two three four five six", line_length=13)
        lines = formatted.splitlines()
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(len(line), 13)

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(format_lyrics(""), "")


if __name__ == "__main__":
    unittest.main()
