#!/usr/bin/env python3
"""Run the UR10e mixed-carton research cell. Outputs never silently overwrite a run."""
import argparse
import json
import logging
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import time
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def inventory(seed,count,order):
    rng=np.random.default_rng(seed)
    if count is None:
        count=int(rng.integers(8,13))
    if not 1<=count<=40:
        raise ValueError('Count must be in 1..40; infeasible batches are reported as PALLET_FULL')
    items=[]
    for i in range(count):
        carton_type=int(rng.integers(0,3))
        if carton_type==0:  # near-square carton
            side=float(rng.uniform(.17,.25))
            d=np.array([side,side*rng.uniform(.88,1.12),rng.uniform(.13,.24)])
        elif carton_type==1:  # long footprint
            d=np.array([rng.uniform(.23,.30),rng.uniform(.15,.22),rng.uniform(.13,.23)])
        else:  # low, broad carton
            d=np.array([rng.uniform(.20,.28),rng.uniform(.18,.26),rng.uniform(.12,.18)])
        items.append({'sku':f'carton-{i:03d}','dimensions':d.tolist(),
                      'carton_type':['near-square','long-footprint','low-broad'][carton_type],
                      'mass':float(rng.uniform(.5,1.8)),
                      'capacity':float(rng.uniform(8.0,14.0))})
    if order=='largest-first':
        items.sort(key=lambda item:np.prod(item['dimensions'][:2]),reverse=True)
    return items


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'config/industrial.yaml')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--count',type=int)
    ap.add_argument('--arrival-order',choices=['random','largest-first','optimized'],default='random',
                    help='physical stream order; optimized is an offline full-manifest baseline')
    ap.add_argument('--buffer-capacity',type=int,help='override configured finite conveyor buffer')
    ap.add_argument('--arrival-interval-s',type=float,help='mean carton inter-arrival interval')
    ap.add_argument('--planner',choices=['beam-search','greedy-safe'],default='beam-search')
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--renderer',choices=['tiny','opengl','egl'],default='tiny')
    ap.add_argument('--speed',type=float,default=4)
    ap.add_argument('--perception',choices=['depth','learned'],default='depth')
    ap.add_argument('--weights',type=Path,default=ROOT/'models/package_seg_runtime_v2.pt')
    ap.add_argument('--device',default='0')
    ap.add_argument('--fault',choices=['none','detection-dropout','vacuum-loss','overweight'],default='none')
    ap.add_argument('--obstacle',action='store_true')
    ap.add_argument('--disturb-box-once',action='store_true',
                    help='test reacquisition by nudging the first box after approach and before descent')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--exit-on-complete',action='store_true')
    args=ap.parse_args()
    if args.output.exists():
        ap.error('Output already exists. Choose a fresh directory to preserve evidence.')
    if not np.isfinite(args.speed) or args.speed<=0:
        ap.error('speed must be positive')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=None;cell=None;code=1;error=None;report={}
    start=time.perf_counter()
    try:
        from palletizing.industrial.configuration import load_industrial_config
        c=load_industrial_config(args.config,ROOT);c['simulation']['seed']=args.seed
        if args.buffer_capacity is not None:c['conveyor']['buffer_capacity']=args.buffer_capacity
        if args.arrival_interval_s is not None:c['conveyor']['arrival_interval_s']=args.arrival_interval_s
        c['conveyor']['planner']=args.planner
        if not 1<=int(c['conveyor']['buffer_capacity'])<=4:
            raise ValueError('GUI conveyor buffer capacity must be in 1..4')
        if not np.isfinite(c['conveyor']['arrival_interval_s']) or c['conveyor']['arrival_interval_s']<=0:
            raise ValueError('Conveyor arrival interval must be finite and positive')
        if c['conveyor']['arrival_interval_s']*(1-c['conveyor']['arrival_jitter_fraction']) < c['conveyor']['feed_duration_s']:
            raise ValueError('Minimum arrival interval must exceed infeed shuttle duration')
        items=inventory(args.seed,args.count,'random' if args.arrival_order=='optimized' else args.arrival_order)
        if args.fault=='overweight':items[0]['mass']=4.0
        if args.arrival_order=='optimized' and args.fault!='overweight':
            from palletizing.industrial.packing import SupportPacker, optimize_batch_order
            packer=SupportPacker(c['pallet']['center_xy'],c['packing']['usable_dimensions'],
                c['pallet']['dimensions'][2],c['packing']['max_height'],c['packing']['gap'],
                c['packing']['support_margin'],c['packing']['max_layers'],c['packing']['max_payload'],
                c['packing']['min_support_fraction'],c['packing']['max_overhang'])
            batch_plan=optimize_batch_order(packer,items,seed=args.seed)
            (args.output/'pick_plan.json').write_text(json.dumps(batch_plan.record(items),indent=2))
            items=[dict(items[i],manifest_index=int(i),pick_index=slot)
                   for slot,i in enumerate(batch_plan.order)]
            logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
            logging.info('Batch plan: strategy=%s cartons=%d layers=%d predicted_height=%.3fm support_min=%.3f utilization=%.1f%%',
                batch_plan.strategy,len(items),batch_plan.layers,batch_plan.max_height,
                batch_plan.minimum_support_fraction,100*batch_plan.utilization)
        (args.output/'config.yaml').write_text(yaml.safe_dump(c))
        (args.output/'inventory.json').write_text(json.dumps(items,indent=2))
        import pybullet as pb
        from pybullet_utils.bullet_client import BulletClient
        from palletizing.rendering import configure_renderer
        from palletizing.industrial.cell import IndustrialCell
        p=BulletClient(connection_mode=pb.DIRECT if args.headless else pb.GUI)
        if not args.headless:
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1)
            logging.info('GUI mouse picking enabled: click and drag cartons to disturb them')
        c['simulation']['camera_renderer_id']=configure_renderer(p,args.renderer,args.headless)
        cell=IndustrialCell(p,c,ROOT,items,args.output,args.perception,args.weights,args.device,
                            not args.headless,args.speed,args.fault,args.obstacle,args.disturb_box_once)
        cell.run();code=0
        logging.info('SUCCESS: %d UR10e placements verified',len(cell.packer.placements))
    except KeyboardInterrupt:
        error='Interrupted';code=130
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        logging.exception('Industrial cell stopped')
    finally:
        if cell is not None:
            if error:
                if p.isConnected():cell.robot.hold()
                cell.enter('FAULT',reason=error)
            report=cell.report()
            if p.isConnected():
                try:
                    from PIL import Image
                    view=p.computeViewMatrixFromYawPitchRoll([.05,.23,.5],2.7,55,-28,0,2)
                    proj=p.computeProjectionMatrixFOV(50,1.5,.05,6)
                    rgba=p.getCameraImage(960,640,view,proj,renderer=pb.ER_TINY_RENDERER)[2]
                    pixels=np.asarray(rgba,dtype=np.uint8)
                    if pixels.size==960*640*4:
                        Image.fromarray(pixels.reshape(640,960,4)[:,:,:3]).save(args.output/'cell.png')
                    else:
                        report['screenshot_error']=f'Unexpected renderer buffer shape: {pixels.shape}'
                except Exception as screenshot_error:
                    report['screenshot_error']=f'{type(screenshot_error).__name__}: {screenshot_error}'
                    logging.exception('Could not save final screenshot; preserving simulation report')
            try:
                cell.robot.planner.close()
            except Exception:
                logging.exception('Could not close the planning client')
        report['environment']={name:importlib.metadata.version(name) for name in ['numpy','scipy','pybullet']}
        report['python']=sys.version
        report['source_sha256']=hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT/'palletizing/industrial').glob('*.py')))).hexdigest()
        report['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
        if args.perception=='learned' and args.weights.is_file():
            report['checkpoint_sha256']=hashlib.sha256(args.weights.read_bytes()).hexdigest()
        report.update(exit_code=code,error=error,wall_seconds=time.perf_counter()-start,
                      perception=args.perception,fault_injection=args.fault,seed=args.seed)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
        if p is not None and p.isConnected():
            if code==0 and not args.headless and not args.exit_on_complete:
                logging.info('Completed. Close window or press Ctrl+C to exit.')
                try:
                    while p.isConnected():time.sleep(.1)
                except KeyboardInterrupt:pass
            if p.isConnected():p.disconnect()
    return code

if __name__=='__main__':
    sys.exit(main())
