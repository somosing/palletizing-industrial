"""Validate the industrial profile before opening a simulator."""
from pathlib import Path
import numpy as np
from ..configuration import load_config


def load_industrial_config(path,root):
    c=load_config(path)
    required={
        'motion':['collision_resolution_rad','clearance_m','planning_timeout_s','rrt_step_rad','rrt_iterations','max_carry_tilt_rad','cartesian_ik_attempts'],
        'packing':['max_height','max_layers','gap','support_margin','max_payload'],
        'process':['settle_seconds','max_tool_payload_kg'],
    }
    for section,keys in required.items():
        for key in keys:
            value=c[section][key]
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{section}.{key} must be finite and positive')
    for section,key in [('motion','rrt_iterations'),('motion','cartesian_ik_attempts'),('packing','max_layers'),('process','detection_retries'),('process','grasp_retries')]:
        value=c[section][key]
        if isinstance(value,bool) or int(value)!=value or value<0:
            raise ValueError(f'{section}.{key} must be a nonnegative integer')
    for key in ['depth_noise_std_m','depth_dropout']:
        if not np.isfinite(c['sensor'][key]) or c['sensor'][key]<0:
            raise ValueError('Invalid sensor noise setting')
    if c['sensor']['depth_dropout']>=1:
        raise ValueError('depth_dropout must be below 1')
    conveyor=c.get('conveyor',{})
    if conveyor.get('enabled',True):
        for key in ('length','width','deck_thickness','feed_duration_s'):
            value=conveyor.get(key)
            if value is None or not np.isfinite(value) or value<=0:
                raise ValueError(f'conveyor.{key} must be finite and positive')
        roller_count=conveyor.get('roller_count',12)
        if isinstance(roller_count,bool) or int(roller_count)!=roller_count or roller_count<2:
            raise ValueError('conveyor.roller_count must be an integer >= 2')
        if conveyor['width'] < max(c['box']['maximum_dimensions'][:2])+0.08:
            raise ValueError('conveyor.width must provide 80 mm clearance around the widest carton')
        capacity=conveyor.get('buffer_capacity',4)
        if isinstance(capacity,bool) or int(capacity)!=capacity or not 1<=capacity<=4:
            raise ValueError('GUI conveyor.buffer_capacity must be an integer in 1..4')
        interval=conveyor.get('arrival_interval_s',2.5)
        jitter=conveyor.get('arrival_jitter_fraction',.25)
        if not np.isfinite(interval) or interval<=0 or not np.isfinite(jitter) or not 0<=jitter<1:
            raise ValueError('Invalid conveyor arrival timing')
        feed_duration=conveyor.get('feed_duration_s',1.2)
        if interval*(1-jitter)<feed_duration:
            raise ValueError('Minimum arrival interval must exceed infeed shuttle duration')
        pitch=conveyor.get('buffer_slot_pitch_m',.47)
        projected=max(c['box']['maximum_dimensions'][:2])*(np.cos(.35)+np.sin(.35))+.025
        if not np.isfinite(pitch) or pitch<projected:
            raise ValueError('Conveyor buffer slot pitch is too small for maximum carton yaw/size')
        first=conveyor.get('buffer_first_slot_offset_x',.5)
        pick_clearance=conveyor.get('pick_station_clearance_m',.10)
        if not np.isfinite(first) or not np.isfinite(pick_clearance) or pick_clearance<0:
            raise ValueError('Invalid conveyor-to-pick-station clearance')
        if first < projected + pick_clearance:
            raise ValueError('Nearest buffer slot overlaps the robot pick/lift envelope')
        center_offset=conveyor.get('center_offset_x',.75)
        last=first+(capacity-1)*pitch+projected/2
        belt_end=center_offset+conveyor['length']/2
        if last>belt_end:
            raise ValueError('Conveyor length does not contain the configured carton buffer')
        feed_start=conveyor.get('feed_start_offset_x',.78)
        if not np.isfinite(feed_start) or feed_start+projected/2>belt_end:
            raise ValueError('Conveyor length does not contain the infeed source carton')
    d=np.asarray(c['packing']['usable_dimensions'])
    if d.shape!=(2,) or not np.isfinite(d).all() or np.any(d<=0) or np.any(d>np.array(c['pallet']['dimensions'][:2])):
        raise ValueError('Packing area must fit within the physical pallet')
    if not 0<c['packing']['min_support_fraction']<=1 or c['packing']['max_overhang']<0:
        raise ValueError('Invalid support policy')
    if not (Path(root)/c['robot']['asset_relative_path']).is_file():
        raise ValueError('Bundled UR10e asset is missing')
    if not np.isclose(c['robot']['tool_length'],.18):
        raise ValueError('Bundled tool length is 0.18 m')
    if not np.allclose(c['camera']['local_translation'],[.1,0,.035]):
        raise ValueError('Camera translation must match the bundled camera housing; edit URDF to change it')
    return c
