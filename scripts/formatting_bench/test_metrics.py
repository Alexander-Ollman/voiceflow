import unittest
from run import score, contains, pcs_guard

class MetricsTests(unittest.TestCase):
    def test_numeric_span_requires_boundary(self):
        self.assertFalse(contains('Set timeout to 120.', '12'))
        self.assertFalse(contains('Set timeout to 12.5.', '12'))
        self.assertFalse(contains('Set timeout to 1.12.', '12'))
        self.assertTrue(contains('Set timeout to 12.', '12'))
    def test_empty_is_failure(self):
        result = score({'input':'Keep this.', 'references':['Keep this.'], 'protected':['Keep']}, '')
        self.assertFalse(result['exact']); self.assertTrue(result['empty'])
        self.assertEqual(result['protected_failures'], ['Keep'])
    def test_guard_rejects_word_replacement_and_negation_deletion(self):
        self.assertEqual(pcs_guard('do not delete', 'Do delete.'), 'do not delete')
        self.assertEqual(pcs_guard('send this now', 'Send this now.'), 'Send this now.')
    def test_acceptable_reference_and_preservation_are_separate(self):
        result = score({'input':'one', 'references':['One.', 'One!'], 'protected':[]}, 'One!')
        self.assertTrue(result['exact']); self.assertTrue(result['changed'])

if __name__ == '__main__': unittest.main()
