#!/usr/bin/env python3
"""Run isolated cell trials, retaining failures and every report/log."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def build_scenarios(seeds,count,arrival_orders,fault_tests=False):
    scenarios=[]
    for order in arrival_orders:
        for seed in seeds:
            name=f'seed_{seed}' if len(arrival_orders)==1 else f'{order}_seed_{seed}'
            scenarios.append((name,seed,count,'none',False,True,order))
    if fault_tests:
        order=arrival_orders[0]
        scenarios.extend([
            ('dropout',7,1,'detection-dropout',False,True,order),
            ('vacuum_loss',7,1,'vacuum-loss',False,False,order),
            ('overweight',7,1,'overweight',False,False,order),
            ('guard_post',7,4,'none',True,True,order),
        ])
    return scenarios


def packing_metrics(report,plan=None):
    placements=report.get('placements',[])
    if placements:
        top=max(float(p['center'][2])+float(p['dimensions'][2])/2 for p in placements)
        base=min(float(p['center'][2])-float(p['dimensions'][2])/2 for p in placements)
        actual_height=top-base
        actual_layers=max(int(p['layer']) for p in placements)+1
    else:
        actual_height=actual_layers=None
    plan=plan or {}
    return {
        'actual_stack_height_m':actual_height,
        'actual_layers':actual_layers,
        'packing_utilization':report.get('packing_volume_utilization'),
        'planned_height_m':plan.get('max_height_m'),
        'planned_layers':plan.get('layer_count'),
        'planned_utilization':plan.get('volume_utilization'),
        'minimum_support_fraction':plan.get('minimum_support_fraction'),
        'search_evaluations':plan.get('search_evaluations'),
    }


def online_metrics(report):
    conveyor=report.get('conveyor_model',{})
    buffer=conveyor.get('buffer',{})
    decisions=conveyor.get('online_pick_decisions',[])
    return {
        'buffer_max_occupancy':buffer.get('max_occupancy'),
        'buffer_blocked_arrivals':buffer.get('blocked_arrivals'),
        'upstream_wait_seconds':buffer.get('upstream_wait_seconds'),
        'online_decisions':len(decisions),
        'unique_cartons_selected':len({d.get('selected_index') for d in decisions}),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,nargs='+',default=[7,11,22])
    ap.add_argument('--count',type=int,default=12)
    ap.add_argument('--arrival-orders',choices=['optimized','largest-first','random'],
                    nargs='+',default=['random'],
                    help='Use random for online trials; other modes are batch baselines')
    ap.add_argument('--buffer-capacity',type=int,default=4,choices=range(1,5))
    ap.add_argument('--arrival-interval-s',type=float,default=2.5)
    ap.add_argument('--planner',choices=['beam-search','greedy-safe'],default='beam-search')
    ap.add_argument('--perception',choices=['depth','learned'],default='depth')
    ap.add_argument('--weights',type=Path,default=ROOT/'models/package_seg_runtime_v2.pt')
    ap.add_argument('--device',default='0')
    ap.add_argument('--fault-tests',action='store_true')
    ap.add_argument('--timeout',type=float,default=600)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Use a new output directory')
    if len(set(args.seeds))!=len(args.seeds):ap.error('--seeds must not contain duplicates')
    if len(set(args.arrival_orders))!=len(args.arrival_orders):ap.error('--arrival-orders must not contain duplicates')
    if args.count<1 or args.count>40:ap.error('--count must be in 1..40')
    if args.arrival_interval_s<=0:ap.error('--arrival-interval-s must be positive')
    args.output.mkdir(parents=True)
    scenarios=build_scenarios(args.seeds,args.count,args.arrival_orders,args.fault_tests)
    rows=[]
    for name,seed,count,fault,obstacle,expected,arrival_order in scenarios:
        folder=args.output/name
        cmd=[sys.executable,str(ROOT/'scripts/run_industrial.py'),'--headless',
             '--seed',str(seed),'--count',str(count),'--fault',fault,'--output',str(folder),
             '--arrival-order',arrival_order,
             '--buffer-capacity',str(args.buffer_capacity),
             '--arrival-interval-s',str(args.arrival_interval_s),
             '--planner',args.planner,
             '--perception',args.perception,'--weights',str(args.weights),'--device',args.device]
        if obstacle:cmd.append('--obstacle')
        with (args.output/f'{name}.log').open('w') as log:
            try:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                returncode=result.returncode
            except subprocess.TimeoutExpired:
                returncode=124
        report_path=folder/'report.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        plan_path=folder/'pick_plan.json'
        plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
        complete=returncode==0 and report.get('state')=='DONE' and report.get('committed')==count
        if expected:
            passed=complete
        else:
            reason={'vacuum-loss':'vacuum loss','overweight':'REJECT_OVERWEIGHT'}[fault]
            passed=(returncode!=0 and report.get('state')=='FAULT' and
                    report.get('committed')==0 and reason in (report.get('error') or ''))
        row={'scenario':name,'seed':seed,'requested':count,'committed':report.get('committed',0),
             'arrival_order':arrival_order,
             'planner':args.planner,'buffer_capacity':args.buffer_capacity,
             'fault':fault,'obstacle':obstacle,
             'expected_complete':expected,'complete':complete,'scenario_pass':passed,
             'wall_seconds':report.get('wall_seconds'),'sim_seconds':report.get('simulated_seconds'),
             **online_metrics(report),
             **packing_metrics(report,plan),
             'peak_vacuum_force_n':report.get('peak_vacuum_force_n'),
             'peak_vacuum_torque_nm':report.get('peak_vacuum_torque_nm'),
             'max_center_error_m':max((v['center_error_m'] for v in report.get('verified_placements',[])),default=None),
             'error':report.get('error') or (f'Exit {returncode}' if returncode else '')}
        rows.append(row);print(json.dumps(row),flush=True)
    normal=[r for r in rows if r['fault']=='none' and not r['obstacle']]
    complete=[r for r in normal if r['complete']]
    wall_times=[r['wall_seconds'] for r in complete if r['wall_seconds'] is not None]
    sim_times=[r['sim_seconds'] for r in complete if r['sim_seconds'] is not None]
    summary={'runs':rows,'nominal_successes':sum(r['complete'] for r in normal),
             'nominal_trials':len(normal),'scenario_passes':sum(r['scenario_pass'] for r in rows),
             'scenario_trials':len(rows),
             'nominal_success_rate':(sum(r['complete'] for r in normal)/len(normal) if normal else 0.0),
             'mean_wall_seconds_completed':(sum(wall_times)/len(wall_times) if wall_times else None),
             'mean_sim_seconds_completed':(sum(sim_times)/len(sim_times) if sim_times else None),
             'buffer_capacity':args.buffer_capacity,'arrival_interval_s':args.arrival_interval_s,
             'planner':args.planner,'perception':args.perception}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (args.output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return 0 if all(r['scenario_pass'] for r in rows) else 1

if __name__=='__main__':sys.exit(main())
