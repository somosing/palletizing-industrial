import unittest
import numpy as np
from palletizing.industrial.packing import SupportPacker, PalletFull


class PackingTests(unittest.TestCase):
    def test_three_layers_have_full_support_and_propagated_load(self):
        p=SupportPacker([0,0],[.31,.31],.12,max_height=1,max_layers=3)
        for d in [[.30,.30,.2],[.25,.25,.2],[.20,.20,.2]]:
            p.commit(p.propose(d,mass=1,capacity=5))
        self.assertEqual([x.layer for x in p.placements],[0,1,2])
        self.assertEqual([x.load_above for x in p.placements],[2,1,0])
        self.assertGreater(p.utilization(),0)
        self.assertLessEqual(p.utilization(),1)
        with self.assertRaises(PalletFull):p.propose([.18,.18,.2])

    def test_bigger_upper_carton_rejected(self):
        p=SupportPacker([0,0],[.31,.31],.12)
        p.commit(p.propose([.29,.29,.2]))
        with self.assertRaises(PalletFull):p.propose([.30,.30,.2])

    def test_crush_and_pallet_mass_limits(self):
        p=SupportPacker([0,0],[.31,.31],.12,max_payload=10)
        p.commit(p.propose([.30,.30,.2],mass=1,capacity=.5))
        with self.assertRaises(PalletFull):p.propose([.20,.20,.2],mass=1)
        q=SupportPacker([0,0],[1,1],.12,max_payload=.5)
        with self.assertRaises(PalletFull):q.propose([.2,.2,.2],mass=1)

    def test_rotation_can_make_carton_fit(self):
        p=SupportPacker([0,0],[.22,.32],.12)
        slot=p.propose([.3,.2,.2])
        self.assertAlmostEqual(slot.yaw,np.pi/2)
        np.testing.assert_allclose(slot.dimensions,[.2,.3,.2])

    def test_adjacent_cartons_do_not_overlap(self):
        p=SupportPacker([0,0],[.8,.7],.12)
        for _ in range(4):p.commit(p.propose([.3,.3,.2]))
        self.assertTrue(all(x.layer==0 for x in p.placements))
        for i,a in enumerate(p.placements):
            for b in p.placements[i+1:]:
                self.assertTrue(np.any(np.abs(np.array(a.center[:2])-b.center[:2])>=.3+p.gap-1e-6))

    def test_invalid_and_duplicate_commit(self):
        p=SupportPacker([0,0],[1,1],.12)
        with self.assertRaises(ValueError):p.propose([np.nan,.2,.2])
        x=p.propose([.2,.2,.2]);p.commit(x)
        with self.assertRaises(ValueError):p.commit(x)


class PlannerTests(unittest.TestCase):
    def test_cartesian_ik_restarts_cover_nearby_elbow_branches(self):
        from palletizing.industrial.motion import CollisionPlanner
        planner=object.__new__(CollisionPlanner)
        planner.robot=type('Robot',(),{
            'low':np.full(6,-6.0),'high':np.full(6,6.0)})()
        planner.rng=np.random.default_rng(5)
        reference=np.zeros(6)
        seeds=planner._ik_seeds(reference,12)
        self.assertEqual(len(seeds),12)
        np.testing.assert_array_equal(seeds[0],reference)
        self.assertTrue(any(seed[2]>.2 for seed in seeds))
        self.assertTrue(any(seed[2]<-.2 for seed in seeds))
        self.assertTrue(any(seed[1]>.1 for seed in seeds))
        self.assertTrue(any(seed[1]<-.1 for seed in seeds))

    def test_rrt_routes_around_barrier(self):
        from palletizing.industrial.motion import CollisionPlanner
        planner=object.__new__(CollisionPlanner)
        planner.rng=np.random.default_rng(5)
        planner.c={'motion':{'planning_timeout_s':3,'rrt_step_rad':.15,'rrt_iterations':1500}}
        planner.resolution=.015
        planner.robot=type('Robot',(),{'low':np.array([-1.,-1.]),'high':np.array([1.,1.])})()
        planner.valid=lambda q: bool(np.all(np.abs(q)<=1) and not(abs(q[0])<.1 and abs(q[1])<.6))
        planner.last_reason='barrier'
        a,b=np.array([-.8,0]),np.array([.8,0])
        self.assertFalse(planner.edge(a,b))
        path=planner.connect(a,b)
        np.testing.assert_allclose(path[0],a);np.testing.assert_allclose(path[-1],b)
        self.assertTrue(all(planner.edge(x,y) for x,y in zip(path[:-1],path[1:])))


class RemovalPlannerTests(unittest.TestCase):
    def test_removal_order_never_removes_a_support_before_its_children(self):
        from palletizing.industrial.removal import plan_removal_order
        placements=[
            {'index':0,'center':[-.3,.5,.2],'layer':0,'support':None},
            {'index':1,'center':[.3,.5,.2],'layer':0,'support':None},
            {'index':2,'center':[-.3,.5,.4],'layer':1,'support':0},
            {'index':3,'center':[-.3,.5,.6],'layer':2,'support':2},
            {'index':4,'center':[.3,.5,.4],'layer':1,'support':1},
        ]
        steps=plan_removal_order(placements,robot_xy=(-.3,.5))
        order=[step.carton_index for step in steps]
        self.assertEqual(set(order),set(range(5)))
        rank={index:i for i,index in enumerate(order)}
        for item in placements:
            if item['support'] is not None:
                self.assertLess(rank[item['index']],rank[item['support']])
        self.assertEqual(order[0],3)

    def test_removal_order_rejects_cycles_and_missing_supports(self):
        from palletizing.industrial.removal import plan_removal_order, RemovalPlanError
        cyclic=[
            {'index':0,'center':[0,0,.2],'layer':0,'support':1},
            {'index':1,'center':[0,0,.4],'layer':1,'support':0},
        ]
        missing=[{'index':0,'center':[0,0,.2],'layer':0,'support':4}]
        with self.assertRaisesRegex(RemovalPlanError,'cycle'):
            plan_removal_order(cyclic)
        with self.assertRaisesRegex(RemovalPlanError,'missing support'):
            plan_removal_order(missing)

    def test_removal_order_fails_closed_when_no_exposed_item_is_reachable(self):
        from palletizing.industrial.removal import plan_removal_order, RemovalPlanError
        placements=[
            {'index':0,'center':[0,0,.2],'layer':0,'support':None},
            {'index':1,'center':[0,0,.4],'layer':1,'support':0},
        ]
        with self.assertRaisesRegex(RemovalPlanError,'No currently exposed carton'):
            plan_removal_order(placements,is_reachable=lambda item: False)

if __name__=='__main__':unittest.main()


class BatchOrderTests(unittest.TestCase):
    @staticmethod
    def make_packer():
        return SupportPacker([-0.05,0.67],[0.8,0.7],0.12,max_height=0.85,
            gap=0.025,support_margin=0.008,max_layers=3,max_payload=80,
            min_support_fraction=0.95,max_overhang=0.02)

    def test_optimizer_returns_complete_deterministic_feasible_sequence(self):
        from scripts.run_industrial import inventory
        from palletizing.industrial.packing import optimize_batch_order
        cartons=inventory(7,12,'random')
        first=optimize_batch_order(self.make_packer(),cartons,seed=7)
        second=optimize_batch_order(self.make_packer(),cartons,seed=7)
        self.assertEqual(first.order,second.order)
        self.assertEqual(sorted(first.order),list(range(len(cartons))))
        self.assertEqual(len(first.placements),12)
        self.assertGreaterEqual(first.layers,1)
        self.assertLessEqual(first.layers,3)
        self.assertEqual(first.search_evaluations,42)
        live=self.make_packer()
        for index in first.order:
            item=cartons[index]
            live.commit(live.propose(item['dimensions'],item['mass'],item['capacity']))
        actual_height=max(p.center[2]+p.dimensions[2]/2 for p in live.placements)-live.base_z
        self.assertAlmostEqual(actual_height,first.max_height,places=6)

    def test_optimizer_preserves_full_support_for_twelve_carton_seed(self):
        from scripts.run_industrial import inventory
        from palletizing.industrial.packing import optimize_batch_order
        cartons=inventory(7,12,'random')
        baseline_items=inventory(7,12,'largest-first')
        baseline=self.make_packer()
        for item in baseline_items:
            baseline.commit(baseline.propose(item['dimensions'],item['mass'],item['capacity']))
        baseline_height=max(p.center[2]+p.dimensions[2]/2 for p in baseline.placements)-baseline.base_z
        chosen=optimize_batch_order(self.make_packer(),cartons,seed=7)
        self.assertLessEqual(chosen.max_height,baseline_height+1e-6)
        self.assertGreaterEqual(chosen.minimum_support_fraction,0.95-1e-6)
        self.assertTrue(all(p.support_fraction>=0.95-1e-6 for p in chosen.placements))

    def test_local_search_is_never_worse_than_its_multistart_seed(self):
        from scripts.run_industrial import inventory
        from palletizing.industrial.packing import optimize_batch_order
        cartons=inventory(7,12,'random')
        packer=self.make_packer()
        multistart=optimize_batch_order(packer,cartons,seed=7,local_search_iterations=0)
        improved=optimize_batch_order(packer,cartons,seed=7,local_search_iterations=32)
        self.assertLessEqual(improved.max_height,multistart.max_height+1e-9)
        self.assertGreater(improved.search_evaluations,multistart.search_evaluations)

    def test_optimizer_finds_feasible_twelve_carton_plans_for_multiple_seeds(self):
        from scripts.run_industrial import inventory
        from palletizing.industrial.packing import optimize_batch_order
        packer=self.make_packer()
        for seed in (7,11,22):
            with self.subTest(seed=seed):
                cartons=inventory(seed,12,'random')
                plan=optimize_batch_order(packer,cartons,seed=seed)
                self.assertEqual(len(plan.order),12)
                self.assertGreaterEqual(plan.minimum_support_fraction,0.95-1e-6)
                self.assertGreater(plan.search_evaluations,1)
