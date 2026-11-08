import unittest
from expanded_latency import stats,model_invoked

class LatencyTests(unittest.TestCase):
    def test_quantiles_include_failed_and_slow_calls(self):
        rows=[dict(latency_ms=n,error=n==20) for n in range(1,21)]
        result=stats(rows)
        self.assertEqual(result['calls'],20)
        self.assertEqual(result['median_ms'],10.5)
        self.assertEqual(result['p95_ms'],19)
        self.assertEqual(result['errors'],1)
    def test_guard_timing_annotation_distinguishes_bypasses(self):
        self.assertFalse(model_invoked('pcs_guarded','Keep userID.'))
        self.assertFalse(model_invoked('pcs_guarded','Hello there.'))
        self.assertTrue(model_invoked('pcs_guarded','hello there'))
        self.assertTrue(model_invoked('pcs','Hello there.'))
        self.assertFalse(model_invoked('rules','hello there'))

    def test_available_memory_guard_keeps_pressure_and_swap_checks(self):
        from watch_resources import guard_reason
        sample=dict(pressure=1,free_bytes=1*2**30,available_bytes=32*2**30)
        self.assertIsNone(guard_reason(sample,memory_metric='available'))
        self.assertIsNotNone(guard_reason(sample))
        self.assertIsNotNone(guard_reason(dict(sample,pressure=2),memory_metric='available'))
        self.assertIsNotNone(guard_reason(sample,swapout_delta=16384,memory_metric='available'))
        self.assertIsNotNone(guard_reason(dict(sample,available_bytes=15*2**30),memory_metric='available'))

if __name__=='__main__':unittest.main()
