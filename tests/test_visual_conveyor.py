import unittest

import numpy as np

from palletizing.industrial.cell import conveyor_interpolate
from palletizing.industrial.configuration import load_industrial_config


class VisualConveyorTests(unittest.TestCase):
    def test_feed_interpolation_starts_ends_and_moves_monotonically(self):
        start=np.array([1.4,-.35,.32])
        stop=np.array([.6,-.35,.32])
        np.testing.assert_allclose(conveyor_interpolate(start,stop,0),start)
        np.testing.assert_allclose(conveyor_interpolate(start,stop,1),stop)
        xs=[conveyor_interpolate(start,stop,t)[0] for t in np.linspace(0,1,11)]
        self.assertTrue(all(a>=b for a,b in zip(xs,xs[1:])))
        self.assertAlmostEqual(conveyor_interpolate(start,stop,.5)[0],.5*(1.4+.6))

    def test_conveyor_dimensions_cover_the_maximum_carton(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        config=load_industrial_config(root/'config/industrial.yaml',root)
        self.assertTrue(config['conveyor']['enabled'])
        self.assertGreaterEqual(config['conveyor']['width'],
                                max(config['box']['maximum_dimensions'][:2])+.08)

    def test_nearest_buffer_slot_clears_pick_and_lift_envelope(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        config=load_industrial_config(root/'config/industrial.yaml',root)
        conveyor=config['conveyor']
        projected=max(config['box']['maximum_dimensions'][:2])*(
            np.cos(.35)+np.sin(.35))+.025
        self.assertGreaterEqual(conveyor['buffer_first_slot_offset_x'],
            projected+conveyor['pick_station_clearance_m'])
        last=conveyor['buffer_first_slot_offset_x']+(
            conveyor['buffer_capacity']-1)*conveyor['buffer_slot_pitch_m']
        belt_end=conveyor['center_offset_x']+conveyor['length']/2
        self.assertGreater(belt_end,last+projected/2)
        self.assertGreater(belt_end,conveyor['feed_start_offset_x']+projected/2)

    def test_rejects_nonfinite_progress(self):
        with self.assertRaises(ValueError):
            conveyor_interpolate([0,0,0],[1,1,1],float('nan'))


if __name__=='__main__':
    unittest.main()
