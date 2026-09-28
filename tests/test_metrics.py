"""The scoring wrapper against hand-computed values under LongProc's official HTML-to-TSV semantics."""
import unittest

from orderkv.metrics import row_f1

GOLD = "name\tvalue\nAlpha One\t10\nBeta, Two\t20\nGamma\t30"


CASES = {
    # Only the text inside the ```tsv fence is scored; punctuation and case are ignored.
    'fenced': ("Here it is:\n```tsv\nname\tvalue\nalpha one\t10\nBeta Two\t20\n```\nDone.", 1.0, 2 / 3),
    # Duplicate predicted rows each count toward precision.
    'duplicates': ("name\tvalue\nGamma\t30\nGamma\t30\nDelta\t40", 2 / 3, 1 / 3),
    # An incomplete last line is dropped.
    'trailing': ("name\tvalue\nAlpha One\t10\nDelta\t40\ntrailing text", 1 / 2, 1 / 3),
    # Without a fence, a preamble line becomes the header and no row can match.
    'preamble': ("Sure:\nname\tvalue\nGamma\t30", 0.0, 0.0),
    'empty': ("", 0.0, 0.0),
}


class OfficialScoreTests(unittest.TestCase):
    def test_hand_computed_scores(self):
        for name, (prediction, precision, recall) in CASES.items():
            with self.subTest(name):
                score = row_f1(prediction, GOLD)
                self.assertAlmostEqual(score['precision'], precision)
                self.assertAlmostEqual(score['recall'], recall)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                self.assertAlmostEqual(score['f1'], f1)


if __name__ == '__main__':
    unittest.main()
