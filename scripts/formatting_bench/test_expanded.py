import json,unittest
from expand import challenges,balanced,V2
from run import contains,ROOT
from expanded_worker import checkpoint_rows
from tempfile import TemporaryDirectory
from pathlib import Path

class ExpandedTests(unittest.TestCase):
    def test_challenge_shape_and_protected_spans(self):
        rows=challenges()
        self.assertEqual(len(rows),357)
        self.assertEqual(len({r['id'] for r in rows}),357)
        for r in rows:
            self.assertTrue(all(contains(r['input'],p) for p in r['protected']),r['id'])
            self.assertTrue(r['references'])
    def test_gold_insertions_keep_annotation_order(self):
        from expand import apply_gold_edits
        self.assertEqual(apply_gold_edits(['Then','left','.'],[(1,1,['I']),(1,1,[','])]),['Then','I',',','left','.'])
    def test_balanced_selection_is_stable_and_unique(self):
        rows=challenges();a=balanced(rows,37);b=balanced(list(reversed(rows)),37)
        self.assertEqual([r['id'] for r in a],[r['id'] for r in b])
        self.assertEqual(len({r['id'] for r in a}),37)
        self.assertEqual(len({r['category'] for r in a}),6)
    def test_resume_discards_only_incomplete_tail(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'results.jsonl';path.write_bytes(b'{"id":"done"}\n{"id":')
            self.assertEqual(checkpoint_rows(path),[{'id':'done'}])
            self.assertEqual(path.read_bytes(),b'{"id":"done"}\n')
    def test_frozen_prefix(self):
        manifest=json.loads((ROOT/'benchmarks/formatting/expanded-v2/selection.json').read_text())
        self.assertEqual(len(manifest['tier_1000_ids']),1000)
        self.assertEqual(len(set(manifest['tier_1000_ids'])),1000)
        self.assertEqual(manifest['tier_1000_ids'][:190],manifest['tier_190_ids'])


    def test_resource_policy_is_conservative(self):
        from watch_resources import guard_reason
        normal=dict(pressure=1,free_bytes=32*2**30)
        self.assertIsNone(guard_reason(normal))
        self.assertIsNotNone(guard_reason(dict(normal,pressure=2)))
        self.assertIsNotNone(guard_reason(dict(normal,free_bytes=15*2**30)))
        self.assertIsNotNone(guard_reason(normal,swapout_delta=16384))
    def test_suite_refuses_low_memory_before_loading(self):
        from unittest.mock import patch
        import expanded_suite
        with patch.object(expanded_suite,'snapshot',return_value=dict(pressure=1,free_bytes=1024,swapout_bytes=0)), \
             patch.object(expanded_suite,'atomic'),patch.object(expanded_suite.subprocess,'Popen') as spawn, \
             patch('sys.argv',['expanded_suite.py','--library','unused']):
            with self.assertRaises(SystemExit):expanded_suite.main()
            spawn.assert_not_called()
    def test_official_errant_adapter_counts_missing_and_false_edits(self):
        import errant
        from expanded_score import errant_wi
        case=dict(id='authored-gold',source='wi-dev',source_tokens='She go home .',
                  gold_m2='S She go home .\nA 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0')
        systems={name:[dict(id=case['id'],output=text)] for name,text in
                 [('unchanged','She go home.'),('correct','She goes home.'),('wrong','She went home.')]}
        metrics,_=errant_wi([case],systems,errant.load('en'))
        self.assertEqual((metrics['correct']['tp'],metrics['correct']['fp'],metrics['correct']['fn']),(1,0,0))
        self.assertEqual((metrics['unchanged']['tp'],metrics['unchanged']['fp'],metrics['unchanged']['fn']),(0,0,1))
        self.assertEqual((metrics['wrong']['tp'],metrics['wrong']['fp'],metrics['wrong']['fn']),(0,1,1))

if __name__=='__main__':unittest.main()
