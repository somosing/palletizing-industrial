import unittest

from scripts.benchmark_industrial import build_scenarios, online_metrics, packing_metrics


class PairedBenchmarkTests(unittest.TestCase):
    def test_builds_paired_orders_for_each_seed(self):
        cases=build_scenarios([7,11],12,['optimized','largest-first'])
        self.assertEqual(len(cases),4)
        self.assertEqual(
            [(case[0],case[1],case[2],case[6]) for case in cases],
            [
                ('optimized_seed_7',7,12,'optimized'),
                ('optimized_seed_11',11,12,'optimized'),
                ('largest-first_seed_7',7,12,'largest-first'),
                ('largest-first_seed_11',11,12,'largest-first'),
            ],
        )

    def test_extracts_actual_and_planned_packing_metrics(self):
        report={
            'placements':[
                {'center':[0,0,.22],'dimensions':[.3,.3,.2],'layer':0},
                {'center':[0,0,.42],'dimensions':[.2,.2,.2],'layer':1},
            ],
            'packing_volume_utilization':.42,
        }
        plan={'max_height_m':.50,'layer_count':2,'volume_utilization':.4,
              'minimum_support_fraction':.98,'search_evaluations':42}
        metrics=packing_metrics(report,plan)
        self.assertAlmostEqual(metrics['actual_stack_height_m'],.4)
        self.assertEqual(metrics['actual_layers'],2)
        self.assertEqual(metrics['packing_utilization'],.42)
        self.assertEqual(metrics['planned_height_m'],.5)
        self.assertEqual(metrics['planned_layers'],2)
        self.assertEqual(metrics['minimum_support_fraction'],.98)

    def test_extracts_online_buffer_and_pick_metrics(self):
        report={'conveyor_model':{
            'buffer':{'max_occupancy':4,'blocked_arrivals':7,'upstream_wait_seconds':12.5},
            'online_pick_decisions':[{'selected_index':2},{'selected_index':0},{'selected_index':2}],
        }}
        metrics=online_metrics(report)
        self.assertEqual(metrics['buffer_max_occupancy'],4)
        self.assertEqual(metrics['buffer_blocked_arrivals'],7)
        self.assertEqual(metrics['upstream_wait_seconds'],12.5)
        self.assertEqual(metrics['online_decisions'],3)
        self.assertEqual(metrics['unique_cartons_selected'],2)


if __name__=='__main__':
    unittest.main()
