import unittest
from shadow_verifier import features

class VerifierTests(unittest.TestCase):
    def test_context_evidence_is_not_automatic_acceptance(self):
        result=features('Use post grass.', 'Use Postgres.', 'Current document: Postgres')
        self.assertEqual(result['edits'][0]['context_supported_words'], ['Postgres'])
        self.assertNotIn('confidence', result)
    def test_negation_and_amount_risks(self):
        result=features('Do not send 120 dollars.', 'Do send 12 dollars.')
        self.assertTrue(result['negation_changed']); self.assertTrue(result['numeric_content_changed'])
    def test_list_structure_preserves_word_order(self):
        result=features('milk eggs bread', '- milk\n- eggs\n- bread')
        self.assertTrue(result['punctuation_or_structure_only'])
        self.assertEqual(result['candidate_list_items'],3)
        self.assertFalse(result['numeric_content_changed'])
    def test_missing_list_item_is_not_just_formatting(self):
        result=features('- milk\n- eggs\n- bread','- milk\n- bread')
        self.assertTrue(result['existing_list_item_count_changed'])
        self.assertFalse(result['punctuation_or_structure_only'])
    def test_identifiers_require_exact_case(self):
        self.assertEqual(features('Use userID.', 'Use user ID.')['identifiers_lost'], ['userID'])
    def test_numbered_list_markers_are_structural(self):
        self.assertFalse(features('milk eggs','1. milk\n2. eggs')['numeric_content_changed'])

if __name__=='__main__': unittest.main()
