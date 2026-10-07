import unittest

import numpy as np

from palletizing.industrial.conveyor import simulate_conveyor_batch
from palletizing.industrial.packing import SupportPacker


def make_packer():
    return SupportPacker([-0.05, 0.67], [0.8, 0.7], 0.12,
                         max_height=0.85, gap=0.025, support_margin=0.008,
                         max_layers=3, max_payload=60,
                         min_support_fraction=0.95, max_overhang=0.02)


def cartons(seed=3, count=8):
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        result.append({
            "sku": f"box-{index}",
            "dimensions": rng.uniform([.10, .10, .10], [.15, .15, .17]).tolist(),
            "mass": 1.0,
            "capacity": 10.0,
        })
    return result


class ConveyorTests(unittest.TestCase):
    def test_physical_buffer_holds_random_arrivals_without_dropping_or_reordering(self):
        from palletizing.industrial.physical_conveyor import FiniteConveyorBuffer
        queue=FiniteConveyorBuffer([0.0,1.0,2.0,3.0,4.0],capacity=2)
        first=queue.advance(1.1)
        self.assertEqual([event.index for event in first],[0,1])
        self.assertEqual(queue.visible,[0,1])
        self.assertEqual(queue.advance(3.1),[])
        self.assertEqual(queue.upstream,[2,3,4])
        self.assertEqual(queue.blocked_arrivals,2)
        released=queue.remove(1)
        self.assertEqual(released,1)
        admitted=queue.advance(3.1)
        self.assertEqual([(event.index,event.slot) for event in admitted],[(2,1)])
        self.assertEqual(queue.visible,[0,2])
        queue.remove(0)
        admitted=queue.advance(4.0)
        self.assertEqual([event.index for event in admitted],[3])
        self.assertEqual(queue.visible,[3,2])
        self.assertEqual(queue.upstream,[4])
        queue.remove(2)
        admitted=queue.advance(4.0)
        self.assertEqual([event.index for event in admitted],[4])
        self.assertEqual(sorted(queue.visible),[3,4])
        self.assertLessEqual(queue.max_occupancy,2)

    def test_random_arrival_schedule_is_reproducible_and_monotonic(self):
        from palletizing.industrial.physical_conveyor import make_arrival_times
        first=make_arrival_times(12,2.5,.25,seed=19)
        second=make_arrival_times(12,2.5,.25,seed=19)
        self.assertEqual(first,second)
        self.assertEqual(first[0],0.0)
        self.assertTrue(all(b>a for a,b in zip(first,first[1:])))

    def test_in_transit_carton_occupies_capacity_but_is_not_pickable(self):
        from palletizing.industrial.physical_conveyor import FiniteConveyorBuffer
        queue=FiniteConveyorBuffer([0.0,1.0],capacity=1)
        admitted=queue.advance(0.0)
        self.assertEqual(queue.visible,[0])
        queue.mark_in_transit(0)
        self.assertEqual(queue.visible,[])
        self.assertEqual(queue.occupancy,1)
        self.assertEqual(queue.advance(2.0),[])
        self.assertEqual(queue.upstream,[1])
        queue.mark_ready(0)
        self.assertEqual(queue.visible,[0])
        self.assertEqual(queue.remove(0),admitted[0].slot)
        self.assertEqual([a.index for a in queue.advance(2.0)],[1])

    def test_seed7_twelve_carton_stream_completes_online_with_current_buffer(self):
        from palletizing.industrial.online_planner import plan_online_pick
        from palletizing.industrial.physical_conveyor import FiniteConveyorBuffer, make_arrival_times
        from scripts.run_industrial import inventory
        items=inventory(seed=7,count=12,order='random')
        rng=np.random.default_rng(7+8803)
        for item in items:
            item['dimensions']=np.maximum(.05,np.asarray(item['dimensions'])+
                rng.normal(0,.0015,3)).tolist()
        queue=FiniteConveyorBuffer(make_arrival_times(12,2.5,.25,7+3907),capacity=4)
        packer=make_packer()
        now=0.0;selected=[]
        while len(packer.placements)<len(items):
            queue.advance(now)
            if not queue.visible:
                now+=.01
                continue
            visible=list(queue.visible)
            plan=plan_online_pick(packer,items,visible,visible,beam_width=24,
                search_depth=min(4,len(visible)),placement_branches=2)
            self.assertIn(plan.first_pick,visible)
            selected.append(plan.first_pick)
            item=items[plan.first_pick]
            packer.commit(packer.propose(item['dimensions'],item['mass'],item['capacity']))
            queue.remove(plan.first_pick)
            now+=10.0
        self.assertEqual(len(selected),12)
        self.assertEqual(len(set(selected)),12)
        self.assertLessEqual(queue.max_occupancy,4)

    def test_beam_search_selects_only_buffered_carton_and_is_deterministic(self):
        from palletizing.industrial.online_planner import plan_online_pick
        items = cartons(seed=7, count=7)
        packer = make_packer()
        first = plan_online_pick(
            packer, items, visible=[0, 1, 2], horizon=[0, 1, 2, 3, 4],
            beam_width=24, search_depth=5, placement_branches=3,
        )
        second = plan_online_pick(
            packer, items, visible=[0, 1, 2], horizon=[0, 1, 2, 3, 4],
            beam_width=24, search_depth=5, placement_branches=3,
        )
        self.assertIn(first.first_pick, [0, 1, 2])
        self.assertEqual(first.sequence, second.sequence)
        self.assertGreater(first.completed_depth, 0)
        self.assertGreater(first.expanded_nodes, 0)
        self.assertEqual(len(first.placements), first.completed_depth)

    def test_beam_search_online_run_respects_buffer_and_completes(self):
        items = cartons(seed=22, count=8)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=22, policy="beam-search",
            buffer_capacity=3, arrival_interval_s=20, pick_service_s=75,
            lookahead_cartons=2, beam_width=24, search_depth=5,
            placement_branches=2,
        )
        self.assertEqual(len(result.placements), len(items))
        self.assertLessEqual(result.max_buffer_occupancy, 3)
        self.assertTrue(all(row["planner"] == "beam-search" for row in result.schedule))
        self.assertTrue(all(row["online_search_nodes"] > 0 for row in result.schedule))
        self.assertTrue(all(row["manifest_index"] in row["visible_manifest_indices"] for row in result.schedule))

    def test_exact_window_certifies_small_finite_problem(self):
        from palletizing.industrial.online_planner import solve_exact_window
        packer = SupportPacker([0, 0], [.31, .31], .12, max_height=.3,
                               gap=.005, support_margin=0, max_layers=1,
                               max_payload=10, min_support_fraction=.95)
        items = [
            {"dimensions": [.1, .1, .1], "mass": 1.0, "capacity": 10.0},
            {"dimensions": [.1, .1, .1], "mass": 1.0, "capacity": 10.0},
        ]
        result = solve_exact_window(
            packer, items, visible=[0, 1], horizon=[0, 1],
            search_depth=2, node_limit=10000,
        )
        self.assertTrue(result.optimality_certified)
        self.assertEqual(result.completed_depth, 2)
        self.assertEqual(result.search_mode, "exact-window")
        self.assertGreater(result.expanded_nodes, 0)

    def test_exact_window_timeout_returns_incumbent_without_false_certificate(self):
        from palletizing.industrial.online_planner import solve_exact_window
        packer = SupportPacker([0, 0], [.31, .31], .12, max_height=.3,
                               gap=.005, support_margin=0, max_layers=1,
                               max_payload=10, min_support_fraction=.95)
        items = [
            {"dimensions": [.1, .1, .1], "mass": 1.0, "capacity": 10.0},
            {"dimensions": [.1, .1, .1], "mass": 1.0, "capacity": 10.0},
        ]
        result = solve_exact_window(
            packer, items, visible=[0, 1], horizon=[0, 1],
            search_depth=2, node_limit=1,
        )
        self.assertFalse(result.optimality_certified)
        self.assertIn(result.first_pick, [0, 1])
        self.assertGreaterEqual(result.completed_depth, 1)

    def test_manifest_joint_beam_uses_full_manifest_and_executes_its_plan(self):
        items = cartons(seed=8, count=4)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=8, policy="manifest-aware",
            buffer_capacity=2, arrival_interval_s=20, pick_service_s=75,
            manifest_planner="joint-beam", manifest_beam_width=8,
            manifest_placement_branches=1,
        )
        self.assertEqual(len(result.placements), 4)
        self.assertIn("joint_beam", result.manifest_plan_strategy)
        self.assertEqual([row["manifest_index"] for row in result.schedule], result.manifest_order)

    def test_manifest_aware_planner_completes_via_declared_upstream_resequencing(self):
        items = cartons(seed=22, count=8)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=22, policy="manifest-aware",
            buffer_capacity=1, arrival_interval_s=20, pick_service_s=75,
            arrival_jitter_fraction=0.15,
        )
        row = result.record()
        self.assertEqual(len(result.placements), len(items))
        self.assertEqual([step["manifest_index"] for step in result.schedule], result.manifest_order)
        self.assertTrue(row["requires_upstream_resequencer"])
        self.assertIsNotNone(result.manifest_plan_strategy)
        self.assertLessEqual(result.max_buffer_occupancy, 1)
        for step in result.schedule:
            self.assertGreaterEqual(step["pick_start_s"], step["arrival_time_s"])

    def test_manifest_aware_waits_for_the_planned_carton_to_arrive(self):
        items = cartons(seed=31, count=6)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=31, policy="manifest-aware",
            buffer_capacity=2, arrival_interval_s=5, pick_service_s=1,
            arrival_jitter_fraction=0.0,
        )
        self.assertEqual(len(result.schedule), len(items))
        for step in result.schedule:
            self.assertGreaterEqual(step["pick_start_s"] + 1e-9, step["arrival_time_s"])

    def test_buffer_is_bounded_and_upstream_backpressure_does_not_drop_cartons(self):
        items = cartons(count=8)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=3, policy="rolling-horizon",
            buffer_capacity=2, arrival_interval_s=1, pick_service_s=10,
            arrival_jitter_fraction=0.0,
        )
        self.assertEqual(len(result.placements), len(items))
        self.assertEqual(len(result.schedule), len(items))
        self.assertLessEqual(result.max_buffer_occupancy, 2)
        self.assertGreater(result.upstream_blocked_arrivals, 0)
        self.assertGreater(result.upstream_wait_seconds, 0)
        self.assertEqual(len({row["manifest_index"] for row in result.schedule}), len(items))

    def test_capacity_one_cannot_reorder_the_arrival_stream(self):
        items = cartons(seed=12, count=6)
        schedules = []
        for policy in ("fifo", "largest-first", "rolling-horizon"):
            result = simulate_conveyor_batch(
                make_packer(), items, seed=12, policy=policy,
                buffer_capacity=1, arrival_interval_s=1, pick_service_s=5,
                lookahead_cartons=2,
                arrival_jitter_fraction=0.0,
            )
            schedules.append([row["manifest_index"] for row in result.schedule])
        self.assertEqual(schedules[0], list(range(6)))
        self.assertEqual(schedules[1], schedules[0])
        self.assertEqual(schedules[2], schedules[0])

    def test_planner_can_select_only_a_carton_that_has_reached_its_buffer(self):
        items = cartons(seed=7, count=7)
        result = simulate_conveyor_batch(
            make_packer(), items, seed=7, policy="rolling-horizon",
            buffer_capacity=3, arrival_interval_s=2, pick_service_s=10,
            lookahead_cartons=3,
            arrival_jitter_fraction=0.0,
        )
        arrived = set()
        for row in result.schedule:
            arrived.update(i for i in range(len(items)) if i * 2 <= row["pick_start_s"] + 1e-9)
            self.assertIn(row["manifest_index"], arrived)
            self.assertLessEqual(len(row["visible_manifest_indices"]), 3)

    def test_sizes_counts_and_runs_are_deterministic(self):
        small = cartons(seed=11, count=8)
        larger = cartons(seed=22, count=12)
        self.assertNotEqual(small[0]["dimensions"], larger[0]["dimensions"])
        args = dict(seed=11, policy="rolling-horizon", buffer_capacity=3,
                    arrival_interval_s=20, pick_service_s=75)
        a = simulate_conveyor_batch(make_packer(), small, **args).record()
        b = simulate_conveyor_batch(make_packer(), small, **args).record()
        self.assertEqual(a["schedule"], b["schedule"])
        self.assertEqual(a["placements"], b["placements"])
        self.assertEqual(a["requested"], 8)
        self.assertEqual(len(simulate_conveyor_batch(
            make_packer(), larger, seed=22, policy="fifo", buffer_capacity=3,
        ).placements), 12)

    def test_rejects_unbounded_or_invalid_buffer_and_timing(self):
        items = cartons(count=2)
        for kwargs in (
            {"buffer_capacity": 0}, {"buffer_capacity": 7},
            {"arrival_interval_s": 0}, {"pick_service_s": -1},
            {"arrival_jitter_fraction": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_conveyor_batch(make_packer(), items, seed=1, **kwargs)


if __name__ == "__main__":
    unittest.main()
