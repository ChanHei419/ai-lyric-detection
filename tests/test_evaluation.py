import tempfile
import unittest
from pathlib import Path

from lyric_detector.evaluation import evaluate_transcriptions

CSV_CONTENT = (
    "audio,reference\n"
    "song1.mp3,hello world\n"
    "song2.mp3,another line here\n"
)


class EvaluateTranscriptionsTests(unittest.TestCase):
    def _write_csv(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        tmp.write(CSV_CONTENT)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_perfect_transcripts_score_zero(self):
        csv_path = self._write_csv()
        summary = evaluate_transcriptions(csv_path, lambda _: "HELLO WORLD! ")
        # Only matches for the first row because the stub returns the same text.
        self.assertEqual(summary["files"], 2)
        self.assertGreater(summary["mean_wer"], 0.0)

    def test_empty_csv_returns_zero_files(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        tmp.write("audio,reference\n")
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)

        summary = evaluate_transcriptions(Path(tmp.name), lambda _: "anything")
        self.assertEqual(summary, {"files": 0, "mean_wer": 0.0})


if __name__ == "__main__":
    unittest.main()
